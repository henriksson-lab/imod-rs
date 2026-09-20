//! Translation of `IMOD/libmesh/mkmesh.c` -- mesh making functions, together
//! with the `Connector` structure and the declarations `IMOD/include/mkmesh.h`
//! pairs with this unit.

use std::cell::Cell;

use crate::imod::libimod::icont::{
    ICONT_CONNECT_BOTTOM, ICONT_CONNECT_INVERT, ICONT_CONNECT_TOP, imod_cont_z_direction,
    imod_contour_get_bbox, imod_contour_nearest,
};
use crate::imod::libimod::imat::{
    B3D_Y, B3D_Z, Imat, imod_mat_delete, imod_mat_new, imod_mat_rot, imod_mat_scale,
    imod_mat_transform,
};
use crate::imod::libimod::imesh::{
    IMESH_MK_SURF, IMOD_MESH_BGNPOLY, IMOD_MESH_END, IMOD_MESH_ENDPOLY, imod_mesh_add_index,
    imod_mesh_add_vert, imod_mesh_new,
};
use crate::imod::libimod::imodel::{ICONT_OPEN, Icont, Imesh, Iobj, Ipoint};
use crate::imod::libimod::iobj::iobj_close;
use crate::imod::libimod::ipoint::{
    imod_point_append, imod_point_area_scale, imod_point_distance, imod_point3d_scale_distance,
};
use crate::imod::libimod::istore::{
    CHANGED_3DWIDTH, CHANGED_COLOR, CHANGED_FCOLOR, CHANGED_TRANS, CHANGED_VALUE1, DrawProps,
    GEN_STORE_CONNECT, GEN_STORE_GAP, GEN_STORE_MINMAX1, GEN_STORE_NOINDEX, GEN_STORE_ONEPOINT,
    GEN_STORE_TRANS, Istore, istore_connect_number, istore_cont_surf_draw_props,
    istore_count_items, istore_default_draw_props, istore_gen_point_items, istore_generate_items,
    istore_get_min_max, istore_insert, istore_list_point_props, istore_point_is_gap,
};

use crate::imod::libcfshr::b3dutil::set_or_clear_flags;

/// Original: `Connector` / `struct connect_struct` (`mkmesh.h:34`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Connector {
    pub b1: i32,
    pub t1: i32,
    pub b2: i32,
    pub t2: i32,
    pub gap: i32,
    pub connect: i32,
    pub skip_to_next: i32,
    pub skip_to_end: i32,
    pub skip_from_start: i32,
    pub skip_index: i32,
}

thread_local! {
    /// Original static `fastmesh` (`mkmesh.c:47`).
    static FASTMESH: Cell<i32> = const { Cell::new(0) };
    /// Original static `meshMin` (`mkmesh.c:48`).
    static MESH_MIN: Cell<Ipoint> = const { Cell::new(Ipoint { x: -1.0e30, y: -1.0e30, z: -1.0e30 }) };
    /// Original static `meshMax` (`mkmesh.c:49`).
    static MESH_MAX: Cell<Ipoint> = const { Cell::new(Ipoint { x: 1.0e30, y: 1.0e30, z: 1.0e30 }) };
    /// Original static `skinFlags` (`mkmesh.c:50`).
    static SKIN_FLAGS: Cell<u32> = const { Cell::new(0) };
}

/// Original: `imeshSetMinMax` (`mkmesh.c:52`).
pub fn imesh_set_min_max(in_min: Ipoint, in_max: Ipoint) {
    MESH_MIN.with(|c| c.set(in_min));
    MESH_MAX.with(|c| c.set(in_max));
}

/// Original: `imeshSetSkinFlags` (`mkmesh.c:57`).
pub fn imesh_set_skin_flags(in_flag: i32, in_fast: i32) {
    SKIN_FLAGS.with(|c| c.set(in_flag as u32));
    FASTMESH.with(|c| c.set(in_fast));
}

/// Original: `imeshContoursCost` (`mkmesh.c:74`).
///
/// Creates and returns a mesh between two contours using a minimum area cost
/// analysis.  Returns `None` for error.
#[allow(clippy::too_many_arguments)]
pub fn imesh_contours_cost(
    obj: &Iobj,
    bc: &mut Icont,
    tc: &mut Icont,
    scale: &Ipoint,
    inside: i32,
    bco: i32,
    tco: i32,
) -> Option<Imesh> {
    let (mut i, mut j, mut k, mut pt, mut jo, mut io): (i32, i32, i32, i32, i32, i32);
    let (mut pt1, mut pt2, mut pt3): (i32, i32, i32) = (0, 0, 0);
    let (mut tlen, mut blen): (i32, i32);
    let mut dofast: i32;
    let (mut bsi, mut tsi): (i32, i32);
    let (mut bsi2, mut tsi2): (i32, i32) = (0, 0);
    let (mut li, mut lj): (i32, i32);
    let (mut iskip, mut jskip): (i32, i32) = (0, 0);
    let (mut endb, mut endt): (i32, i32);
    let mut jobase: i32;
    let mut mincost: f64;
    let mut ccost: f64;
    let mut minp = Ipoint::default();
    let mut maxp = Ipoint::default();
    let mut valmin = 0.0f32;
    let mut valmax = 0.0f32;
    let mut obj_props = DrawProps::default();
    let mut bc_props = DrawProps::default();
    let mut tc_props = DrawProps::default();
    let mut pt_props = DrawProps::default();
    let mut bc_state: i32 = 0;
    let mut tc_state: i32 = 0;
    let mut surf_state: i32 = 0;
    let mut state3: i32 = 0;
    let mut store3_top = false;
    let mut props3_top = false;
    let mut trans_max: i32;
    let (mut trans1, mut trans2, mut trans3): (i32, i32, i32);
    let mut state_test = CHANGED_COLOR | CHANGED_FCOLOR | CHANGED_3DWIDTH;
    let mut num_con: i32 = 0;
    let mut cur_con: i32 = 0;
    let mut next_con: i32 = -1;
    let mut started_at_con = 0;
    let mut starts_connected = 0;
    let mut ends_connected = 0;
    let mut new_bstore: i32 = 0;
    let mut new_tstore: i32 = 0;
    let mut bc_store: Option<Vec<Istore>> = None;
    let mut tc_store: Option<Vec<Istore>> = None;

    /* Index [0] is bottom contour, Index [1] is top contour. */
    let mut si = [0i32; 2]; /* start index */
    let mut direction = [0i32; 2]; /* contour direction. */
    let mut si_copy = [0i32; 2];

    /* Check input data. */
    if bc.pts.is_empty() || tc.pts.is_empty() {
        return None;
    }

    /* Init internal data. */
    let mut mesh = imod_mesh_new()?.remove(0);
    let mut maxsize: i32 = 0;
    if istore_get_min_max(
        &obj.store,
        obj.cont.len() as i32,
        GEN_STORE_MINMAX1,
        &mut valmin,
        &mut valmax,
    ) != 0
    {
        state_test |= CHANGED_VALUE1;
    }

    /* Contours from an open contour object should be marked as open? */
    let open_obj = if iobj_close(obj.flags) != 0 { 0 } else { 1 };
    let tsize = tc.pts.len() as i32;
    let mut tdim = tsize;
    let bsize = bc.pts.len() as i32;
    let mut bdim = bsize;

    /* Single or 2-point close contours cannot be meshed to each other; two
    single-point open contours cannot be meshed */
    if (open_obj == 0 && tdim < 3 && bdim < 3) || (open_obj != 0 && tdim < 2 && bdim < 2) {
        return None;
    }

    /* Reduce extent of meshing for open objects */
    if open_obj != 0 {
        tdim -= 1;
        bdim -= 1;
    }
    let csize = ((tdim + 1) * (bdim + 1)) as usize;
    let totind = csize - 1;

    let mut up: Vec<f32> = vec![0.; (bsize * tsize) as usize];
    let mut down: Vec<f32> = vec![0.; (tsize * bsize) as usize];
    let mut cost: Vec<f64> = vec![0.; csize];
    let mut path: Vec<u8> = vec![0; csize];
    let mut path2: Vec<u8> = vec![0; csize];

    /* Get default drawing properties for each contour, and flags for what is
    already changed from default. */
    istore_default_draw_props(obj, &mut obj_props);
    i = istore_cont_surf_draw_props(
        &obj.store,
        &obj_props,
        &mut bc_props,
        bco,
        bc.surf,
        &mut bc_state,
        &mut surf_state,
    );
    j = istore_cont_surf_draw_props(
        &obj.store,
        &obj_props,
        &mut tc_props,
        tco,
        tc.surf,
        &mut tc_state,
        &mut surf_state,
    );
    let skin_flags = SKIN_FLAGS.with(|c| c.get());
    if skin_flags & IMESH_MK_SURF == 0 {
        bc_state = i;
        tc_state = j;
    }

    let any_trans = (bc_state & CHANGED_TRANS)
        | (tc_state & CHANGED_TRANS)
        | istore_count_items(&bc.store, GEN_STORE_TRANS, 1)
        | istore_count_items(&tc.store, GEN_STORE_TRANS, 1);

    /* Find direction of each contour. */
    direction[0] = imod_cont_z_direction(Some(bc));
    direction[1] = imod_cont_z_direction(Some(tc));
    if direction[0] == 0 {
        direction[0] = 1;
    }
    if direction[1] == 0 {
        direction[1] = 1;
    }

    si[0] = 0;
    si[1] = 0;
    dofast = 1;

    /* Add gaps for open contours in closed obj and make connectors at gaps */
    if manage_gaps(
        bc,
        &mut bc_store,
        &mut new_bstore,
        tc,
        &mut tc_store,
        &mut new_tstore,
        iobj_close(obj.flags),
        &direction,
        scale,
    ) != 0
    {
        cleanup_icc_arrays(
            &mut mesh, &mut up, &mut down, &mut cost, &mut path, &mut path2, bc, new_bstore,
            bc_store, tc, new_tstore, tc_store, num_con, None,
        );
        return None;
    }

    /* Set flags for whether the contours are open one way or another */
    let b_open = i32::from(open_obj != 0 || istore_point_is_gap(&bc.store, bsize - 1) != 0);
    let t_open = i32::from(open_obj != 0 || istore_point_is_gap(&tc.store, tsize - 1) != 0);

    /* Get connectors if any */
    let connects = make_connectors(
        bc,
        tc,
        &mut num_con,
        iobj_close(obj.flags),
        direction[0] * direction[1],
    );
    let mut connects: Vec<Connector> = connects.unwrap_or_default();
    cur_con = 0;
    let lc = (num_con - 1) as usize;

    /* Set starting index for both contours. */
    if open_obj != 0 {
        dofast = FASTMESH.with(|c| c.get());

        /* If the OBJECT TYPE IS OPEN CONTOUR, then ignore the computed contour
        directions; set directions to +, and invert the second one if that makes
        them match up better */
        direction[1] = 1;
        direction[0] = 1;

        /* But first see if starts or ends are connected and use that to set the
        polarity.  Also set polarity if there is more than 1 connector */
        if num_con != 0 {
            starts_connected = i32::from(
                (connects[0].b1 == 0 && (connects[0].t1 == 0 || connects[0].t2 == tsize - 1))
                    || connects[0].skip_from_start != 0,
            );
            ends_connected = i32::from(
                (connects[lc].b2 == bsize - 1
                    && (connects[lc].t1 == 0 || connects[lc].t2 == tsize - 1))
                    || connects[lc].skip_to_end != 0,
            );
        }

        if starts_connected != 0 || ends_connected != 0 || num_con > 1 {
            if (starts_connected != 0
                && (connects[0].t2 == tsize - 1
                    || (connects[0].t1 == 0
                        && connects[0].skip_from_start != 0
                        && connects[0].skip_index > connects[0].b2)))
                || (ends_connected != 0
                    && (connects[lc].t1 == 0
                        || (connects[lc].b2 == bsize - 1
                            && connects[lc].skip_to_end != 0
                            && connects[lc].skip_index < connects[lc].t1)))
                || (num_con > 1 && connects[1].t1 < connects[0].t1)
            {
                direction[1] = -1;
            }
        } else {
            /* If there is no polarity info, now use endpoint distances */
            let dista = imod_point_distance(&bc.pts[0], &tc.pts[0])
                + imod_point_distance(&bc.pts[(bsize - 1) as usize], &tc.pts[(tsize - 1) as usize]);
            let distb = imod_point_distance(&bc.pts[0], &tc.pts[(tsize - 1) as usize])
                + imod_point_distance(&bc.pts[(bsize - 1) as usize], &tc.pts[0]);
            if distb < dista {
                direction[1] = -1;
            }
        }

        /* If either one is marked as inverted before and is NOT inverted this
        time, invert them both */
        if ((bc.flags & ICONT_CONNECT_INVERT) != 0 && (bc.flags & ICONT_CONNECT_BOTTOM) != 0)
            || ((tc.flags & ICONT_CONNECT_TOP) != 0
                && direction[1]
                    != (if (tc.flags & ICONT_CONNECT_INVERT) != 0 {
                        -1
                    } else {
                        1
                    }))
        {
            direction[0] *= -1;
            direction[1] *= -1;
        }

        /* Then set the invert flag for whichever is now inverted */
        set_or_clear_flags(&mut bc.flags, ICONT_CONNECT_INVERT, 1 - direction[0]);
        set_or_clear_flags(&mut tc.flags, ICONT_CONNECT_INVERT, 1 - direction[1]);

        /* Invert the direction of connections if bottom is reversed */
        if num_con != 0 {
            invert_connectors(&mut connects, num_con, &direction);
        }

        /* Invert the starting points for reversed directions and (re-)evaluate
        if starts or ends are connected */
        if direction[0] < 0 {
            si[0] = bsize - 1;
        }
        if direction[1] < 0 {
            si[1] = tsize - 1;
        }

        if num_con != 0 {
            starts_connected = i32::from(
                (connects[0].b1 == si[0] && connects[0].t1 == si[1])
                    || connects[0].skip_from_start != 0,
            );
            ends_connected = i32::from(
                (connects[lc].b2 == bsize - 1 - si[0] && connects[lc].t2 == tsize - 1 - si[1])
                    || connects[lc].skip_to_end != 0,
            );

            /* Finally, if the starts are connected, use those to set indexes */
            if starts_connected != 0 {
                si[0] = connects[0].b1;
                si[1] = connects[0].t1;
                cur_con = 1;
            }
        }
    } else {
        /* CLOSED OBJECT: if connectors, start at the first connector and set flag,
        but reverse connectors if bottom is inverted */
        if num_con != 0 {
            invert_connectors(&mut connects, num_con, &direction);
            si[0] = connects[0].b2;
            si[1] = connects[0].t2;
            started_at_con = 1;
            cur_con = 1;
        } else if b_open != 0 && t_open == 0 {
            si[1] = imod_contour_nearest(Some(tc), &bc.pts[0]);
        } else if t_open != 0 && b_open == 0 {
            si[0] = imod_contour_nearest(Some(bc), &tc.pts[0]);
        } else {
            /* Both closed contours, no connectors */
            dofast = FASTMESH.with(|c| c.get());

            /* Try to have all mesh start at about the same place so it looks
            better with fake transparency. */
            imod_contour_get_bbox(Some(bc), &mut minp, &mut maxp);
            si[0] = imod_contour_nearest(Some(bc), &minp);
            bsi2 = imod_contour_nearest(Some(bc), &maxp);
            if direction[0] < 0 {
                bsi2 = bsize - 1 - bsi2;
            }

            /* Now find a similar point in the top contour. */

            imod_contour_get_bbox(Some(tc), &mut minp, &mut maxp);
            si[1] = imod_contour_nearest(Some(tc), &minp);
            tsi2 = imod_contour_nearest(Some(tc), &maxp);
            if direction[1] < 0 {
                tsi2 = tsize - 1 - tsi2;
            }
        }
    }

    /*
     * Build the matrices of areas for up and down triangles
     */
    build_area_matrices(
        bc,
        direction[0],
        tc,
        direction[1],
        scale,
        &mut up,
        &mut down,
        open_obj,
    );

    while cur_con <= num_con - ends_connected {
        blen = bdim;
        tlen = tdim;

        if cur_con <= 0 || connects[(cur_con - 1) as usize].skip_to_next == 0 {
            /* If there are connectors, need to revise the endpoints */
            if num_con != 0 {
                /* index for endpoints is the current connector index, unless at the
                end.  In that case it is the first connector or the other end of
                both open contours */
                next_con = cur_con;
                if cur_con >= num_con {
                    next_con = if started_at_con != 0 { 0 } else { -1 };
                }
                if next_con >= 0 {
                    endb = connects[next_con as usize].b1;
                    endt = connects[next_con as usize].t1;
                } else {
                    endb = if direction[0] > 0 { bsize - 1 } else { 0 };
                    endt = if direction[1] > 0 { tsize - 1 } else { 0 };
                }

                /* get the extent to mesh; routine will mesh 0 to dim inclusive.
                If both open contours, length can be zero and zero must not wrap */
                blen = if direction[0] > 0 {
                    endb - si[0]
                } else {
                    si[0] - endb
                };
                if blen < 0 || (open_obj == 0 && blen == 0) {
                    blen += bsize;
                }
                tlen = if direction[1] > 0 {
                    endt - si[1]
                } else {
                    si[1] - endt
                };
                if tlen < 0 || (open_obj == 0 && tlen == 0) {
                    tlen += tsize;
                }
            }

            /* flip starting indexes if directions are reversed */
            bsi = si[0];
            if direction[0] < 0 {
                bsi = bsize - 1 - bsi;
            }
            tsi = si[1];
            if direction[1] < 0 {
                tsi = tsize - 1 - tsi;
            }

            /* Get the cost and path matrices for these starting points */

            cost_from_area_matrices(
                &up, &down, &mut cost, &mut path, bdim, tdim, bsize, bsi, tsi, blen, tlen, -1.0,
            );

            /* optimize connections only if there are no connectors */
            if num_con == 0 {
                if dofast != 0 {
                    if b_open == 0 && t_open == 0 {
                        /* If both closed contours, first try the opposite corner and
                        replace path if better */
                        ccost = cost[totind];
                        cost_from_area_matrices(
                            &up, &down, &mut cost, &mut path2, bdim, tdim, bsize, bsi2, tsi2, blen,
                            tlen, ccost,
                        );
                        if cost[totind] < ccost {
                            bsi = bsi2;
                            tsi = tsi2;
                            ccost = cost[totind];
                            path.copy_from_slice(&path2);
                        }

                        /* Now go halfway around to find a new starting point, and use
                        that instead if it is better */
                        i = bdim;
                        j = tdim;
                        for _step in 0..(bdim + tdim) / 2 {
                            if path[(i + j * (bdim + 1)) as usize] != 0 {
                                i -= 1;
                            } else {
                                j -= 1;
                            }
                        }

                        bsi2 = (bsi + i) % bsize;
                        tsi2 = (tsi + j) % tsize;
                        cost_from_area_matrices(
                            &up, &down, &mut cost, &mut path2, bdim, tdim, bsize, bsi2, tsi2, bdim,
                            tdim, ccost,
                        );
                        if cost[totind] < ccost {
                            bsi = bsi2;
                            tsi = tsi2;
                            path.copy_from_slice(&path2);
                        }
                    }
                } else {
                    /* Time-consuming: find starting point that gives minimum area */

                    mincost = cost[totind];

                    if t_open != 0 || b_open != 0 {
                        /* For open contours, just do reverse direction */

                        build_area_matrices(
                            bc,
                            direction[0],
                            tc,
                            -direction[1],
                            scale,
                            &mut up,
                            &mut down,
                            open_obj,
                        );
                        cost_from_area_matrices(
                            &up, &down, &mut cost, &mut path, bdim, tdim, bsize, bsi, tsi, bdim,
                            tdim, mincost,
                        );

                        /* If it's better, flip direction; otherwise rebuild areas */
                        if cost[totind] < mincost {
                            direction[1] *= -1;
                        } else {
                            build_area_matrices(
                                bc,
                                direction[0],
                                tc,
                                direction[1],
                                scale,
                                &mut up,
                                &mut down,
                                open_obj,
                            );
                        }
                    } else {
                        /* Loop on starting points in top */
                        for i in 0..tsize {
                            cost_from_area_matrices(
                                &up, &down, &mut cost, &mut path, bdim, tdim, bsize, bsi, i, bdim,
                                tdim, mincost,
                            );

                            ccost = cost[totind];

                            if ccost < mincost {
                                mincost = ccost;
                                tsi = i;
                            }
                        }
                    }

                    /* Redo path from final starting indices and direction */
                    cost_from_area_matrices(
                        &up, &down, &mut cost, &mut path, bdim, tdim, bsize, bsi, tsi, bdim, tdim,
                        -1.0,
                    );
                }
            }

            /* get back the starting indexes in terms of unreversed data */
            if direction[0] < 0 {
                bsi = bsize - 1 - bsi;
            }
            if direction[1] < 0 {
                tsi = tsize - 1 - tsi;
            }

            if mesh.vert.is_empty() {
                /*
                 * first time through, copy contour points to mesh vert array
                 */
                si_copy[0] = bsi;
                si_copy[1] = tsi;
                let vsize = (bsize + tsize + 2) as usize;
                mesh.vert = vec![Ipoint::default(); vsize];

                /* copy bottom contour data to mesh array. */
                pt = si_copy[0];
                for i in 0..bsize {
                    mesh.vert[i as usize] = bc.pts[pt as usize];
                    pt += direction[0];
                    if pt == bsize {
                        pt = 0;
                        iskip = i;
                    } else if pt < 0 {
                        pt = bsize - 1;
                        iskip = i;
                    }
                }
                mesh.vert[bsize as usize] = bc.pts[si_copy[0] as usize];

                /* copy top contour data to mesh vert array. */
                pt = si_copy[1];
                i = bsize + 1;
                while i < vsize as i32 - 1 {
                    mesh.vert[i as usize] = tc.pts[pt as usize];
                    pt += direction[1];
                    if pt == tsize {
                        pt = 0;
                        jskip = i - (bsize + 1);
                    } else if pt < 0 {
                        pt = tsize - 1;
                        jskip = i - (bsize + 1);
                    }
                    i += 1;
                }
                mesh.vert[vsize - 1] = tc.pts[si_copy[1] as usize];
            }

            /* The connection loop.  Set starting indexes and offsets  */
            i = blen;
            j = tlen;
            io = (direction[0] * (bsi - si_copy[0]) + bsize) % bsize;
            jobase = (direction[1] * (tsi - si_copy[1]) + tsize) % tsize;
            jo = jobase + bsize + 1;

            /* Do we need to add a triangle at the terminal connector? */
            if num_con != 0 && next_con >= 0 && connects[next_con as usize].gap == 0 {
                if connects[next_con as usize].b2 != connects[next_con as usize].b1 {
                    i += 1;
                    path[(i + j * (bdim + 1)) as usize] = 1;
                } else if connects[next_con as usize].t2 != connects[next_con as usize].t1 {
                    j += 1;
                    path[(i + j * (bdim + 1)) as usize] = 0;
                }
            }

            /* Now make the mesh by following the path. */
            while i != 0 || j != 0 {
                k = mesh.list.len() as i32;
                if path[(i + j * (bdim + 1)) as usize] != 0 {
                    li = i - 1;
                    pt3 = (bsi + direction[0] * i + bsize) % bsize;
                    pt2 = (bsi + direction[0] * li + bsize) % bsize;
                    pt1 = (tsi + direction[1] * j + tsize) % tsize;
                    if !(((bc.flags & ICONT_OPEN) != 0 && li + io == iskip)
                        || istore_point_is_gap(&bc.store, if direction[0] > 0 { pt2 } else { pt3 })
                            != 0
                        || outside_mesh_limits(
                            &tc.pts[pt1 as usize],
                            &bc.pts[pt2 as usize],
                            &bc.pts[pt3 as usize],
                        ) != 0)
                    {
                        chunk_add_triangle(
                            &mut mesh,
                            j + jo,
                            li + io,
                            i + io,
                            &mut maxsize,
                            inside,
                        );
                        state3 = bc_state;
                        props3_top = false;
                        store3_top = false;
                    }
                    i -= 1;
                } else {
                    lj = j - 1;
                    pt3 = (tsi + direction[1] * j + tsize) % tsize;
                    pt1 = (tsi + direction[1] * lj + tsize) % tsize;
                    pt2 = (bsi + direction[0] * i + bsize) % bsize;
                    if !(((tc.flags & ICONT_OPEN) != 0 && lj + jobase == jskip)
                        || istore_point_is_gap(&tc.store, if direction[1] > 0 { pt1 } else { pt3 })
                            != 0
                        || outside_mesh_limits(
                            &tc.pts[pt1 as usize],
                            &bc.pts[pt2 as usize],
                            &tc.pts[pt3 as usize],
                        ) != 0)
                    {
                        chunk_add_triangle(
                            &mut mesh,
                            lj + jo,
                            i + io,
                            j + jo,
                            &mut maxsize,
                            inside,
                        );
                        state3 = tc_state;
                        props3_top = true;
                        store3_top = true;
                    }
                    j -= 1;
                }

                if mesh.list.len() as i32 > k {
                    if k == 0 {
                        k += 1;
                    }
                    let k2 = if inside != 0 { k + 2 } else { k + 1 };
                    let k3 = if inside != 0 { k + 1 } else { k + 2 };
                    if (tc_state & state_test) != 0 || !tc.store.is_empty() {
                        istore_gen_point_items(
                            &tc.store,
                            &tc_props,
                            tc_state,
                            pt1,
                            &mut mesh.store,
                            k,
                            state_test,
                        );
                    }
                    let store3_empty = if store3_top {
                        tc.store.is_empty()
                    } else {
                        bc.store.is_empty()
                    };
                    if (state3 & state_test) != 0 || !store3_empty {
                        let store3 = if store3_top { &tc.store } else { &bc.store };
                        let props3 = if props3_top { &tc_props } else { &bc_props };
                        istore_gen_point_items(
                            store3,
                            props3,
                            state3,
                            pt3,
                            &mut mesh.store,
                            k3,
                            state_test,
                        );
                    }
                    if (bc_state & state_test) != 0 || !bc.store.is_empty() {
                        istore_gen_point_items(
                            &bc.store,
                            &bc_props,
                            bc_state,
                            pt2,
                            &mut mesh.store,
                            k2,
                            state_test,
                        );
                    }

                    /* Handle trans states - if any point has a positive trans, set all
                    trans to at least 1 so triangle is recognized as trans */
                    if any_trans != 0 {
                        let p1_state =
                            istore_list_point_props(&tc.store, &tc_props, &mut pt_props, pt1);
                        trans1 = pt_props.trans;
                        trans_max = pt_props.trans;
                        let p2_state =
                            istore_list_point_props(&bc.store, &bc_props, &mut pt_props, pt2);
                        trans2 = pt_props.trans;
                        trans_max = trans_max.max(pt_props.trans);
                        let store3 = if store3_top { &tc.store } else { &bc.store };
                        let props3 = if props3_top { &tc_props } else { &bc_props };
                        let p3_state = istore_list_point_props(store3, props3, &mut pt_props, pt3);
                        trans3 = pt_props.trans;
                        trans_max = trans_max.max(pt_props.trans);
                        if (tc_state | bc_state | p1_state | p2_state | p3_state) & CHANGED_TRANS
                            != 0
                        {
                            if trans_max != 0 && trans1 == 0 {
                                trans1 = 1;
                            }
                            if trans_max != 0 && trans2 == 0 {
                                trans2 = 1;
                            }
                            if trans_max != 0 && trans3 == 0 {
                                trans3 = 1;
                            }
                            pt_props.trans = trans1;
                            istore_generate_items(
                                &mut mesh.store,
                                &pt_props,
                                CHANGED_TRANS,
                                k,
                                CHANGED_TRANS,
                            );
                            pt_props.trans = trans2;
                            istore_generate_items(
                                &mut mesh.store,
                                &pt_props,
                                CHANGED_TRANS,
                                k2,
                                CHANGED_TRANS,
                            );
                            pt_props.trans = trans3;
                            istore_generate_items(
                                &mut mesh.store,
                                &pt_props,
                                CHANGED_TRANS,
                                k3,
                                CHANGED_TRANS,
                            );
                        }
                    }
                }
            }
        }
        /* Set starting indexes for next round */
        if cur_con < num_con {
            si[0] = connects[cur_con as usize].b2;
            si[1] = connects[cur_con as usize].t2;
        }
        cur_con += 1;
    }

    cleanup_icc_arrays(
        &mut mesh,
        &mut up,
        &mut down,
        &mut cost,
        &mut path,
        &mut path2,
        bc,
        new_bstore,
        bc_store,
        tc,
        new_tstore,
        tc_store,
        num_con,
        Some(connects),
    );

    if !mesh.list.is_empty() {
        chunk_mesh_add_index(&mut mesh, IMOD_MESH_ENDPOLY, &mut maxsize);
    }
    imod_mesh_add_index(&mut mesh, IMOD_MESH_END);
    Some(mesh)
}

/// Original static `manageGaps` (`mkmesh.c:669`).
///
/// Makes sure gaps get connected correctly if possible; inserts a gap at the
/// end of open contours in closed objects and connects endpoints of gaps that
/// match up.
#[allow(clippy::too_many_arguments)]
pub fn manage_gaps(
    bc: &mut Icont,
    bc_store: &mut Option<Vec<Istore>>,
    new_bstore: &mut i32,
    tc: &mut Icont,
    tc_store: &mut Option<Vec<Istore>>,
    new_tstore: &mut i32,
    obj_closed: i32,
    direction: &[i32; 2],
    scale: &Ipoint,
) -> i32 {
    let mut store = Istore::default();
    let min_gap_ratio = 1.0f32;
    let mut pntb = Ipoint::default();
    let mut pntt = Ipoint::default();
    let mut num_gaps = [0i32; 2];
    let mut big_open = [0i32; 2];
    let mut retval = 0;
    let (mut tpt, mut tnext, mut bpt, mut bnext): (i32, i32, i32, i32) = (0, 0, 0, 0);
    let (mut tgmin, mut bgmin): (i32, i32) = (0, 0);
    let mut max_con_num: i32;
    let mut j: i32 = 0;
    let mut end_connected = 0;
    let (mut dist, mut distmin, mut maxgap): (f32, f32, f32);

    num_gaps[0] = istore_count_items(&bc.store, GEN_STORE_GAP, 0);
    num_gaps[1] = istore_count_items(&tc.store, GEN_STORE_GAP, 0);
    let no_gaps = i32::from(num_gaps[0] + num_gaps[1] == 0);
    store.type_ = GEN_STORE_GAP;
    store.flags = GEN_STORE_ONEPOINT;
    store.value.set_i(0);

    /* If either contour is open in closed object, insert a gap at the end */
    if obj_closed != 0
        && (bc.flags & ICONT_OPEN) != 0
        && istore_point_is_gap(&bc.store, bc.pts.len() as i32 - 1) == 0
    {
        if dup_store_if_needed(bc, bc_store, new_bstore) != 0 {
            return 1;
        }
        store.index.set_i(bc.pts.len() as i32 - 1);
        if istore_insert(&mut bc.store, store) != 0 {
            return 1;
        }
        num_gaps[0] += 1;
    }
    if obj_closed != 0
        && (tc.flags & ICONT_OPEN) != 0
        && istore_point_is_gap(&tc.store, tc.pts.len() as i32 - 1) == 0
    {
        if dup_store_if_needed(tc, tc_store, new_tstore) != 0 {
            return 1;
        }
        store.index.set_i(tc.pts.len() as i32 - 1);
        if istore_insert(&mut tc.store, store) != 0 {
            return 1;
        }
        num_gaps[1] += 1;
    }

    /* If there are gaps in only one, or neither, no more to do */
    if num_gaps[1] == 0 || num_gaps[0] == 0 {
        return 0;
    }

    /* Make list of all the gaps and find maximum connection number */
    let mut bot_list: Vec<i32> = vec![0; num_gaps[0] as usize];
    let mut top_list: Vec<i32> = vec![0; num_gaps[1] as usize];
    max_con_num = 0;
    for bt in 0..2usize {
        num_gaps[bt] = 0;
        big_open[bt] = -1;
        let cont: &Icont = if bt == 0 { bc } else { tc };
        for i in 0..cont.store.len() {
            let storep = cont.store[i];
            if storep.type_ == GEN_STORE_GAP && storep.index.i() >= 0 {
                /* Get real start of gap, add it to list if it is not on it already */
                bnext = ends_of_whole_gap(cont, storep.index.i(), &mut bpt);
                let bt_list: &mut Vec<i32> = if bt == 0 {
                    &mut bot_list
                } else {
                    &mut top_list
                };
                let mut jj: i32 = 0;
                while jj < num_gaps[bt] {
                    if bt_list[jj as usize] == bpt {
                        break;
                    }
                    jj += 1;
                }
                if jj == num_gaps[bt] {
                    bt_list[num_gaps[bt] as usize] = bpt;
                    if bnext < bpt && (bnext > 0 || bpt < cont.pts.len() as i32 - 1) {
                        big_open[bt] = num_gaps[bt];
                    }
                    num_gaps[bt] += 1;
                }
            }
            if storep.type_ == GEN_STORE_CONNECT {
                max_con_num = max_con_num.max(storep.value.i());
            }
        }
    }
    let mut bot_left = num_gaps[0];
    let mut top_left = num_gaps[1];

    /* Connect multi-point gaps across contour ends */
    if big_open[0] >= 0 && big_open[1] >= 0 {
        bpt = bot_list[big_open[0] as usize];
        tpt = top_list[big_open[1] as usize];
        if add_connector_if_none(
            bc,
            bc_store,
            new_bstore,
            bpt,
            tc,
            tc_store,
            new_tstore,
            tpt,
            &mut max_con_num,
            direction,
        ) != 0
        {
            return 1;
        }
        end_connected = 1;
        bot_list[big_open[0] as usize] = -1;
        top_list[big_open[1] as usize] = -1;
        bot_left -= 1;
        top_left -= 1;
    }

    /* Loop on finding closest pairs of gaps and connecting them */
    while bot_left != 0 && top_left != 0 {
        /* Look at all pairs of gaps and find one with closest midpoint */
        distmin = 1.0e30;
        for bgap in 0..num_gaps[0] {
            if bot_list[bgap as usize] < 0 {
                continue;
            }
            bpt = bot_list[bgap as usize];
            bnext = ends_of_whole_gap(bc, bpt, &mut j);
            pntb.x = (bc.pts[bpt as usize].x + bc.pts[bnext as usize].x) / 2.;
            pntb.y = (bc.pts[bpt as usize].y + bc.pts[bnext as usize].y) / 2.;
            pntb.z = (bc.pts[bpt as usize].z + bc.pts[bnext as usize].z) / 2.;
            for tgap in 0..num_gaps[1] {
                if top_list[tgap as usize] < 0 {
                    continue;
                }
                tpt = top_list[tgap as usize];
                tnext = ends_of_whole_gap(tc, tpt, &mut j);
                pntt.x = (tc.pts[tpt as usize].x + tc.pts[tnext as usize].x) / 2.;
                pntt.y = (tc.pts[tpt as usize].y + tc.pts[tnext as usize].y) / 2.;
                pntt.z = (tc.pts[tpt as usize].z + tc.pts[tnext as usize].z) / 2.;
                dist = imod_point3d_scale_distance(&pntb, &pntt, scale);
                if dist < distmin {
                    tgmin = tgap;
                    bgmin = bgap;
                    distmin = dist;
                }
            }
        }

        /* Find maximum of two gap extents */
        bpt = bot_list[bgmin as usize];
        bnext = ends_of_whole_gap(bc, bpt, &mut j);
        tpt = top_list[tgmin as usize];
        tnext = ends_of_whole_gap(tc, tpt, &mut j);
        maxgap = imod_point_distance(&bc.pts[bpt as usize], &bc.pts[bnext as usize]);
        dist = imod_point_distance(&tc.pts[tpt as usize], &tc.pts[tnext as usize]);
        maxgap = maxgap.max(dist);

        /* Connect the two gaps if they are close relative to the size of the gaps,
        or if they are open contours openings and there are no other gaps,
        and if none of the four points are already connected somewhere */
        if distmin < min_gap_ratio * maxgap || no_gaps != 0 {
            if add_connector_if_none(
                bc,
                bc_store,
                new_bstore,
                bpt,
                tc,
                tc_store,
                new_tstore,
                tpt,
                &mut max_con_num,
                direction,
            ) != 0
            {
                retval = 1;
                break;
            }

            /* Keep track of whether an endpoint has been connected */
            if bpt == bc.pts.len() as i32 - 1 || tpt == tc.pts.len() as i32 - 1 {
                end_connected = 1;
            }
        }

        /* Remove these two gaps from lists regardless of whether connected */
        bot_list[bgmin as usize] = -1;
        top_list[tgmin as usize] = -1;
        bot_left -= 1;
        top_left -= 1;
    }

    /* If both were open, make sure at least one end if connected to something,
    otherwise connect the two openings */
    if obj_closed != 0
        && (bc.flags & ICONT_OPEN) != 0
        && (tc.flags & ICONT_OPEN) != 0
        && end_connected == 0
    {
        let bpt = bc.pts.len() as i32 - 1;
        let tpt = tc.pts.len() as i32 - 1;
        retval = add_connector_if_none(
            bc,
            bc_store,
            new_bstore,
            bpt,
            tc,
            tc_store,
            new_tstore,
            tpt,
            &mut max_con_num,
            direction,
        );
    }
    retval
}

/// Original static `addConnectorIfNone` (`mkmesh.c:867`).
///
/// Adds a pair of connectors at the ends of a shared gap if the user has not
/// inserted any connectors.
#[allow(clippy::too_many_arguments)]
pub fn add_connector_if_none(
    bc: &mut Icont,
    bc_store: &mut Option<Vec<Istore>>,
    new_bstore: &mut i32,
    bpt: i32,
    tc: &mut Icont,
    tc_store: &mut Option<Vec<Istore>>,
    new_tstore: &mut i32,
    tpt: i32,
    max_con_num: &mut i32,
    direction: &[i32; 2],
) -> i32 {
    let mut cpt: i32 = 0;
    let bnext = ends_of_whole_gap(bc, bpt, &mut cpt);
    let tnext = ends_of_whole_gap(tc, tpt, &mut cpt);
    let mut store = Istore::default();

    if istore_connect_number(&bc.store, bpt) < 0
        && istore_connect_number(&bc.store, bnext) < 0
        && istore_connect_number(&tc.store, tpt) < 0
        && istore_connect_number(&tc.store, tnext) < 0
    {
        store.type_ = GEN_STORE_CONNECT;
        store.flags = GEN_STORE_ONEPOINT;
        store.value.set_i(0);
        for cpt in 0..2 {
            *max_con_num += 1;
            store.value.set_i(*max_con_num);
            store.index.set_i(if cpt != 0 { bnext } else { bpt });
            if dup_store_if_needed(bc, bc_store, new_bstore) != 0
                || istore_insert(&mut bc.store, store) != 0
            {
                return 1;
            }
            store.index.set_i(if cpt != 0 { tnext } else { tpt });
            if direction[0] != direction[1] {
                store.index.set_i(if cpt != 0 { tpt } else { tnext });
            }
            if dup_store_if_needed(tc, tc_store, new_tstore) != 0
                || istore_insert(&mut tc.store, store) != 0
            {
                return 1;
            }
        }
    }
    0
}

/// Original static `endsOfWholeGap` (`mkmesh.c:905`).
///
/// Finds the point at the ends of a gap, passing over gap points.
pub fn ends_of_whole_gap(cont: &Icont, pt: i32, new_start: &mut i32) -> i32 {
    let psize = cont.pts.len() as i32;
    let mut next = pt;
    let mut i = 0;
    *new_start = pt;
    while i < psize / 2 {
        next = if next != 0 { next - 1 } else { psize - 1 };
        if istore_point_is_gap(&cont.store, next) == 0 {
            break;
        }
        *new_start = next;
        i += 1;
    }

    next = (pt + 1) % psize;
    i = 0;
    while i < psize / 2 && istore_point_is_gap(&cont.store, next) != 0 {
        next = (next + 1) % psize;
        i += 1;
    }
    next
}

/// Original static `dupStoreIfNeeded` (`mkmesh.c:928`).
///
/// Duplicates the contour store if it has not been done already, and keeps
/// track of the old store and a flag of whether this was done.
pub fn dup_store_if_needed(
    cont: &mut Icont,
    old_store: &mut Option<Vec<Istore>>,
    made_new: &mut i32,
) -> i32 {
    if *made_new != 0 {
        return 0;
    }
    *old_store = Some(cont.store.clone());
    *made_new = 1;
    0
}

/// Original static `cleanupICCarrays` (`mkmesh.c:941`).
///
/// Cleans up all arrays from the contour cost routine.  The C `free` calls on
/// `up`/`down`/`cost`/`path`/`path2`/`connects` become drops of the owned
/// vectors; the contour store restore is the part that is observable.  As in
/// the source, `mesh` is a parameter this routine does not touch.
#[allow(clippy::too_many_arguments)]
pub fn cleanup_icc_arrays(
    _mesh: &mut Imesh,
    up: &mut Vec<f32>,
    down: &mut Vec<f32>,
    cost: &mut Vec<f64>,
    path: &mut Vec<u8>,
    path2: &mut Vec<u8>,
    bc: &mut Icont,
    new_bstore: i32,
    bc_store: Option<Vec<Istore>>,
    tc: &mut Icont,
    new_tstore: i32,
    tc_store: Option<Vec<Istore>>,
    num_con: i32,
    connects: Option<Vec<Connector>>,
) {
    if num_con != 0 {
        drop(connects);
    }
    path.clear();
    path2.clear();
    cost.clear();
    up.clear();
    down.clear();
    if new_bstore != 0 {
        bc.store = bc_store.unwrap_or_default();
    }
    if new_tstore != 0 {
        tc.store = tc_store.unwrap_or_default();
    }
}

/// Original static `build_area_matrices` (`mkmesh.c:966`).
#[allow(clippy::too_many_arguments)]
pub fn build_area_matrices(
    bc: &Icont,
    bdir: i32,
    tc: &Icont,
    tdir: i32,
    scale: &Ipoint,
    up: &mut [f32],
    down: &mut [f32],
    open_obj: i32,
) {
    let mut zscale = 0.0f32;
    if (scale.x - scale.y).abs() < 1.0e-5 * scale.x {
        zscale = scale.z / scale.x;
    }

    let bsize = bc.pts.len() as i32;
    let tsize = tc.pts.len() as i32;
    let bpt = &bc.pts;
    let tpt = &tc.pts;
    let mut bdim = bsize;
    let mut tdim = tsize;
    if open_obj != 0 {
        bdim -= 1;
        tdim -= 1;
    }

    let mut i = 0;
    if bdir < 0 {
        i = bsize - 1;
    }
    let mut j = 0;
    if tdir < 0 {
        j = tsize - 1;
    }

    for l in 0..tsize {
        let mut nj = j + tdir;
        if nj == tsize {
            nj = 0;
        }
        if nj < 0 {
            nj = tsize - 1;
        }
        let lbase = l * bsize;
        if zscale != 0. {
            for k in 0..bsize {
                /* 1/14/18: In ctffind, two tests were significantly faster than
                ni = (i + bdir + bsize) % bsize; */
                let mut ni = i + bdir;
                if ni == bsize {
                    ni = 0;
                }
                if ni < 0 {
                    ni = bsize - 1;
                }
                if k == bdim {
                    up[(k + lbase) as usize] = 0.;
                } else {
                    up[(k + lbase) as usize] = point_area_quick(
                        &tpt[j as usize],
                        &bpt[i as usize],
                        &bpt[ni as usize],
                        zscale,
                    );
                }
                if l == tdim {
                    down[(k + lbase) as usize] = 0.;
                } else {
                    down[(k + lbase) as usize] = point_area_quick(
                        &tpt[j as usize],
                        &tpt[nj as usize],
                        &bpt[i as usize],
                        zscale,
                    );
                }
                i = ni;
            }
        } else {
            for k in 0..bsize {
                let mut ni = i + bdir;
                if ni == bsize {
                    ni = 0;
                }
                if ni < 0 {
                    ni = bsize - 1;
                }
                if k == bdim {
                    up[(k + lbase) as usize] = 0.;
                } else {
                    up[(k + lbase) as usize] = imod_point_area_scale(
                        &tpt[j as usize],
                        &bpt[i as usize],
                        &bpt[ni as usize],
                        scale,
                    );
                }
                if l == tdim {
                    down[(k + lbase) as usize] = 0.;
                } else {
                    down[(k + lbase) as usize] = imod_point_area_scale(
                        &tpt[j as usize],
                        &tpt[nj as usize],
                        &bpt[i as usize],
                        scale,
                    );
                }
                i = ni;
            }
        }
        j = nj;
    }
}

/// Original static `pointAreaQuick` (`mkmesh.c:1041`).
///
/// A quicker function for area that uses only the zscale, computes the cross
/// product without a function call, uses `sqrtf`, and leaves off the factor of
/// 0.5.
pub fn point_area_quick(p1: &Ipoint, p2: &Ipoint, p3: &Ipoint, zscale: f32) -> f32 {
    let mut n = Ipoint::default();
    let mut n1 = Ipoint::default();
    let mut n2 = Ipoint::default();

    n1.x = p1.x - p2.x;
    n1.y = p1.y - p2.y;
    n1.z = (p1.z - p2.z) * zscale;
    n2.x = p3.x - p2.x;
    n2.y = p3.y - p2.y;
    n2.z = (p3.z - p2.z) * zscale;
    n.x = (n1.y * n2.z) - (n1.z * n2.y);
    n.y = (n1.z * n2.x) - (n1.x * n2.z);
    n.z = (n1.x * n2.y) - (n1.y * n2.x);

    (n.x * n.x + n.y * n.y + n.z * n.z).sqrt()
}

/// Original static `cost_from_area_matrices` (`mkmesh.c:1063`).
///
/// Computes the minimum area path to every possible connection, thus allowing
/// one to follow a minimum area path from ending to starting connection.
#[allow(clippy::too_many_arguments)]
pub fn cost_from_area_matrices(
    up: &[f32],
    down: &[f32],
    cost: &mut [f64],
    path: &mut [u8],
    bdim: i32,
    tdim: i32,
    bsize: i32,
    sb: i32,
    st: i32,
    bmax: i32,
    tmax: i32,
    curmin: f64,
) {
    let mut jl: i32 = 0;
    let mut il: i32 = 0;
    let mut rowmin: f64;
    let (mut costup, mut costdown): (f64, f64);

    cost[0] = 0.;
    let mut j = st;
    if j == tdim {
        j = 0;
    }
    rowmin = 0.0;
    for l in 0..=tmax {
        let lbase = l * (bdim + 1);
        let mut i = sb;
        if i == bdim {
            i = 0;
        }
        for k in 0..=bmax {
            let ind = (k + lbase) as usize;
            if k == 0 {
                /* If in first column, add area of down triangles */
                if l != 0 {
                    cost[ind] =
                        cost[ind - (bdim + 1) as usize] + down[(i + jl * bsize) as usize] as f64;
                    path[ind] = 0;
                    rowmin = cost[ind];
                }
            } else if l == 0 {
                /* If in first row, add area of up triangles */
                cost[ind] = cost[ind - 1] + up[(il + j * bsize) as usize] as f64;
                path[ind] = 1;
            } else {
                /* Otherwise figure out which is smaller and add area
                and save path direction for that */
                costdown = cost[ind - (bdim + 1) as usize] + down[(i + jl * bsize) as usize] as f64;
                costup = cost[ind - 1] + up[(il + j * bsize) as usize] as f64;

                if costdown < costup {
                    cost[ind] = costdown;
                    path[ind] = 0;
                } else {
                    cost[ind] = costup;
                    path[ind] = 1;
                }
                /* keep track of minimum along the row */
                if cost[ind] < rowmin {
                    rowmin = cost[ind];
                }
            }

            /* adjust row index into area matrices */
            il = i;
            i += 1;
            if i == bdim {
                i = 0;
            }
        }

        /* adjust column index into area matrices */
        jl = j;
        j += 1;
        if j == tdim {
            j = 0;
        }

        /* If there is a current minimum and row min exceeds it, abort */
        if curmin >= 0.0 && rowmin > curmin {
            cost[(bmax + tmax * (bdim + 1)) as usize] = rowmin;
            return;
        }
    }
}

/// Original: `makeConnectors` (`mkmesh.c:1136`).
///
/// Analyzes contour stores for connectors and encodes them in an array of
/// structures with the bottom and top indices.  Call with `dir_product` 0 to
/// look for a single connection between contours on the same plane.
pub fn make_connectors(
    bc: &Icont,
    tc: &Icont,
    num_con: &mut i32,
    closed_obj: i32,
    dir_product: i32,
) -> Option<Vec<Connector>> {
    let mut min_bridge_num = 0;
    if let Ok(value) = std::env::var("MIN_BRIDGE_CONNECT_NUM") {
        min_bridge_num = atoi(&value);
    }

    *num_con = 0;
    let mut max_con = istore_count_items(&bc.store, GEN_STORE_CONNECT, 0);
    let max_top = istore_count_items(&tc.store, GEN_STORE_CONNECT, 0);
    max_con = max_con.min(max_top);
    if max_con == 0 {
        return None;
    }
    let mut conn: Vec<Connector> = Vec::new();

    let open_obj = if closed_obj != 0 { 0 } else { 1 };
    let tsize = tc.pts.len() as i32;
    let mut open_dir: i32 = 0;

    for i in 0..bc.store.len() {
        let stp = bc.store[i];
        if stp.type_ == GEN_STORE_CONNECT {
            if dir_product == 0 && stp.value.i() < min_bridge_num {
                continue;
            }

            /* First see if the connect # is used already */
            let mut used = 0;
            for j in 0..*num_con as usize {
                if stp.value.i() == conn[j].connect {
                    used = 1;
                }
            }
            if used != 0 {
                continue;
            }

            /* Next look for connect in top contour */
            used = 0;
            let mut stp2 = Istore::default();
            for j in 0..tc.store.len() {
                let cand = tc.store[j];
                if cand.type_ == GEN_STORE_CONNECT && cand.value.i() == stp.value.i() {
                    used = 1;
                    stp2 = cand;
                    break;
                }
            }

            if used == 0 {
                continue;
            }

            /* Check for direction consistency */
            if *num_con != 0 {
                if dir_product == 0 {
                    return None;
                }
                let start = conn[0].t1;
                let last = conn[(*num_con - 1) as usize].t1;
                let mid = stp2.index.i();

                /* No longer need to consider open contours of closed objects */
                if closed_obj != 0 {
                    /* For closed contours, direction is set, any two points are OK but
                    the next point must be between the last and the end. */
                    if *num_con > 1
                        && (dir_product * (mid - last) + tsize) % tsize
                            + (dir_product * (start - mid) + tsize) % tsize
                            != (dir_product * (start - last) + tsize) % tsize
                    {
                        continue;
                    }
                } else {
                    /* For open object, direction 1 can flip so any two points are OK,
                    but next point must change in same direction as the last */
                    if *num_con == 2 {
                        open_dir = if last - start > 0 { 1 } else { -1 };
                    }
                    if (mid - last) * open_dir < 0 {
                        continue;
                    }
                }
            }

            /* Define the connector */
            let mut connp = Connector {
                b1: stp.index.i(),
                t1: stp2.index.i(),
                b2: stp.index.i(),
                t2: stp2.index.i(),
                gap: 0,
                connect: stp.value.i(),
                skip_to_next: 0,
                skip_to_end: 0,
                skip_from_start: 0,
                skip_index: 0,
            };

            /* Look for explicit neighboring 3rd point in bottom */
            let mut used = 0;
            let mut k = 0usize;
            while used == 0 && k < bc.store.len() {
                let stp = bc.store[k];
                if stp.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                    break;
                }
                if stp.type_ == GEN_STORE_CONNECT && stp.value.i() == connp.connect {
                    if stp.index.i() == connp.b1 + 1 {
                        connp.b2 += 1;
                        used = 1;
                    }

                    /* If connector is at zero and this is an end of cont, back off
                    the starting index to the end of the cont */
                    if stp.index.i() == bc.pts.len() as i32 - 1 && connp.b1 == 0 && closed_obj != 0
                    {
                        connp.b1 = bc.pts.len() as i32 - 1;
                        used = 1;
                    }
                }
                k += 1;
            }

            /* Look for explicit neighboring 3rd point in top */
            let mut k = 0usize;
            while used == 0 && k < tc.store.len() {
                let stp = tc.store[k];
                if stp.flags & (GEN_STORE_NOINDEX | 3) != 0 {
                    break;
                }
                if stp.type_ == GEN_STORE_CONNECT && stp.value.i() == connp.connect {
                    if stp.index.i() == connp.t1 + 1 {
                        connp.t2 += 1;
                        used = 1;
                    }
                    if stp.index.i() == tc.pts.len() as i32 - 1 && connp.t1 == 0 && closed_obj != 0
                    {
                        connp.t1 = tc.pts.len() as i32 - 1;
                        used = 1;
                    }
                }
                k += 1;
            }

            /* If a third point was found, check for gap */
            if (connp.b1 != connp.b2 && istore_point_is_gap(&bc.store, connp.b1) != 0)
                || (connp.t1 != connp.t2 && istore_point_is_gap(&tc.store, connp.t1) != 0)
            {
                connp.gap = 1;
            }

            conn.push(connp);
            *num_con += 1;
        }
    }

    /* Return now with single connector between contours on same plane */
    if dir_product == 0 {
        if *num_con == 0 {
            return None;
        }
        return Some(conn);
    }

    /* Now that all connectors are gotten, check for gaps before and after
    each point and introduce an implied third point on other side of the
    gap as long as the point does not appear in another connector */
    for j in 0..*num_con as usize {
        /* Skip if there are already three points and it is NOT a gap */
        if conn[j].gap == 0 && (conn[j].b1 != conn[j].b2 || conn[j].t1 != conn[j].t2) {
            continue;
        }

        /* Look forward on bottom. */
        let mut index = (conn[j].b1 + 1) % bc.pts.len() as i32;
        if conn[j].b1 == conn[j].b2
            && istore_point_is_gap(&bc.store, conn[j].b1) != 0
            && !(conn[j].b1 == bc.pts.len() as i32 - 1 && open_obj != 0)
        {
            let mut used = 0;
            for i in 0..*num_con as usize {
                if i != j && (conn[i].b1 == index || conn[i].b2 == index) {
                    used = 1;
                }
            }
            if used == 0
                && ((closed_obj != 0 && istore_connect_number(&bc.store, index) >= 0)
                    || istore_connect_number(&bc.store, index) == conn[j].connect)
            {
                conn[j].b2 = index;
                conn[j].gap = 1;
            }
        }

        /* Look backward on bottom */
        index = (bc.pts.len() as i32 + conn[j].b1 - 1) % bc.pts.len() as i32;
        if conn[j].b1 == conn[j].b2
            && istore_point_is_gap(&bc.store, index) != 0
            && !(index == bc.pts.len() as i32 - 1 && open_obj != 0)
        {
            let mut used = 0;
            for i in 0..*num_con as usize {
                if i != j && (conn[i].b1 == index || conn[i].b2 == index) {
                    used = 1;
                }
            }
            if used == 0
                && ((closed_obj != 0 && istore_connect_number(&bc.store, index) >= 0)
                    || istore_connect_number(&bc.store, index) == conn[j].connect)
            {
                conn[j].b1 = index;
                conn[j].gap = 1;
            }
        }

        /* Look forward on top */
        index = (conn[j].t1 + 1) % tc.pts.len() as i32;
        if conn[j].t1 == conn[j].t2
            && istore_point_is_gap(&tc.store, conn[j].t1) != 0
            && !(conn[j].t1 == tc.pts.len() as i32 - 1 && open_obj != 0)
        {
            let mut used = 0;
            for i in 0..*num_con as usize {
                if i != j && (conn[i].t1 == index || conn[i].t2 == index) {
                    used = 1;
                }
            }
            if used == 0
                && ((closed_obj != 0 && istore_connect_number(&tc.store, index) >= 0)
                    || istore_connect_number(&tc.store, index) == conn[j].connect)
            {
                conn[j].t2 = index;
                conn[j].gap = 1;
            }
        }

        /* Look backward on top */
        index = (tc.pts.len() as i32 + conn[j].t1 - 1) % tc.pts.len() as i32;
        if conn[j].t1 == conn[j].t2
            && istore_point_is_gap(&tc.store, index) != 0
            && !(index == tc.pts.len() as i32 - 1 && open_obj != 0)
        {
            let mut used = 0;
            for i in 0..*num_con as usize {
                if i != j && (conn[i].t1 == index || conn[i].t2 == index) {
                    used = 1;
                }
            }
            if used == 0
                && ((closed_obj != 0 && istore_connect_number(&tc.store, index) >= 0)
                    || istore_connect_number(&tc.store, index) == conn[j].connect)
            {
                conn[j].t1 = index;
                conn[j].gap = 1;
            }
        }
    }

    /* Now look for skip segments between connectors or at start/end */
    let open_dir = if closed_obj != 0 {
        dir_product
    } else {
        open_dir
    };
    for j in 0..*num_con as usize {
        let nj = (j + 1) % *num_con as usize;
        if j == 0 && open_obj != 0 {
            /* Test for skip at start of open contours */
            /* First test that top is at known or possible end and blocker is
            before the bottom point */
            if (conn[j].b1 > 0 && conn[j].t1 == 0 && open_dir >= 0)
                || (conn[j].t2 == tc.pts.len() as i32 - 1 && open_dir <= 0)
            {
                let connum1 = istore_connect_number(&bc.store, conn[j].b1 - 1);
                if connum1 >= 0 && connum1 != conn[j].connect {
                    conn[j].skip_from_start = 1;
                    conn[j].skip_index = conn[j].b1 - 1;
                }
            }

            /* Then test if bottom is at end and top has block in right direction */
            if conn[j].b1 == 0 && conn[j].t1 > 0 && conn[j].t2 < tc.pts.len() as i32 - 1 {
                if open_dir >= 0 {
                    let connum1 = istore_connect_number(&tc.store, conn[j].t1 - 1);
                    if connum1 >= 0 && connum1 != conn[j].connect {
                        conn[j].skip_from_start = 1;
                        conn[j].skip_index = conn[j].t1 - 1;
                    }
                }
                if conn[j].skip_from_start == 0 && open_dir <= 0 {
                    let connum1 = istore_connect_number(&tc.store, conn[j].t2 + 1);
                    if connum1 >= 0 && connum1 != conn[j].connect {
                        conn[j].skip_from_start = 1;
                        conn[j].skip_index = conn[j].t2 + 1;
                    }
                }
            }
        }

        if j < (*num_con - 1) as usize || closed_obj != 0 {
            /* Test for skip between two successive contours */
            /* First look for pair of blockers on bottom */
            let mut index = (conn[j].b2 + 1) % bc.pts.len() as i32;
            let mut connum1 = istore_connect_number(&bc.store, index);
            index = (bc.pts.len() as i32 + conn[nj].b1 - 1) % bc.pts.len() as i32;
            let mut connum2 = istore_connect_number(&bc.store, index);
            if connum1 >= 0
                && connum2 >= 0
                && connum1 != conn[j].connect
                && connum1 != conn[nj].connect
                && connum2 != conn[j].connect
                && connum2 != conn[nj].connect
            {
                conn[j].skip_to_next = 1;
            }

            /* Look for pair on top, trickier because of direction dependence */
            if conn[j].skip_to_next == 0 {
                if dir_product > 0 {
                    index = (conn[j].t2 + 1) % tc.pts.len() as i32;
                } else {
                    index = (tc.pts.len() as i32 + conn[j].t1 - 1) % tc.pts.len() as i32;
                }
                connum1 = istore_connect_number(&tc.store, index);
                if dir_product < 0 {
                    index = (conn[nj].t2 + 1) % tc.pts.len() as i32;
                } else {
                    index = (tc.pts.len() as i32 + conn[nj].t1 - 1) % tc.pts.len() as i32;
                }
                connum2 = istore_connect_number(&tc.store, index);
                if connum1 >= 0
                    && connum2 >= 0
                    && connum1 != conn[j].connect
                    && connum1 != conn[nj].connect
                    && connum2 != conn[j].connect
                    && connum2 != conn[nj].connect
                {
                    conn[j].skip_to_next = 1;
                }
            }
        } else if j != 0 || conn[j].skip_from_start == 0 {
            /* Test for skip to end of open contours */
            /* First test that top is at known or possible end and blocker is
            after the bottom point */
            if (conn[j].b2 < bc.pts.len() as i32 - 1 && conn[j].t1 == 0 && open_dir <= 0)
                || (conn[j].t2 == tc.pts.len() as i32 - 1 && open_dir >= 0)
            {
                let connum1 = istore_connect_number(&bc.store, conn[j].b2 + 1);
                if connum1 >= 0 && connum1 != conn[j].connect {
                    conn[j].skip_to_end = 1;
                    conn[j].skip_index = conn[j].b2 + 1;
                }
            }

            /* Then test if bottom is at end and top has block in right direction */
            if conn[j].b2 == bc.pts.len() as i32 - 1
                && conn[j].t1 > 0
                && conn[j].t2 < tc.pts.len() as i32 - 1
            {
                if open_dir >= 0 {
                    let connum1 = istore_connect_number(&tc.store, conn[j].t2 + 1);
                    if connum1 >= 0 && connum1 != conn[j].connect {
                        conn[j].skip_to_end = 1;
                        conn[j].skip_index = conn[j].t2 + 1;
                    }
                }
                if conn[j].skip_to_end == 0 && open_dir <= 0 {
                    let connum1 = istore_connect_number(&tc.store, conn[j].t1 - 1);
                    if connum1 >= 0 && connum1 != conn[j].connect {
                        conn[j].skip_to_end = 1;
                        conn[j].skip_index = conn[j].t1 - 1;
                    }
                }
            }
        }
    }
    Some(conn)
}

/// Original static `invertConnectors` (`mkmesh.c:1443`).
///
/// If bottom direction is negative, this reverses the order of the connectors
/// in the array and exchanges b1 and b2.  If top direction is negative, it
/// exchanges t1 and t2.
pub fn invert_connectors(connects: &mut [Connector], num_con: i32, direction: &[i32; 2]) {
    let num_con = num_con as usize;
    if direction[0] < 0 {
        /* Swap connectors for bottom inversion */
        for i in 0..num_con / 2 {
            connects.swap(i, num_con - 1 - i);
        }

        /* Fix up skips at start and end */
        let tmp = connects[0].skip_to_end;
        if connects[num_con - 1].skip_from_start != 0 {
            connects[num_con - 1].skip_to_end = 1;
            connects[num_con - 1].skip_from_start = 0;
        }
        if tmp != 0 {
            connects[0].skip_from_start = 1;
            connects[0].skip_to_end = 0;
        }

        /* Swap b1 and b2, and fix skip to next */
        let zero_skip = connects[0].skip_to_next;
        connects[0].skip_to_next = 0;
        for i in 0..num_con {
            std::mem::swap(&mut connects[i].b1, &mut connects[i].b2);
            let tmp = (i + 1) % num_con;
            if tmp != 0 {
                if connects[tmp].skip_to_next != 0 {
                    connects[tmp].skip_to_next = 0;
                    connects[i].skip_to_next = 1;
                }
            } else {
                connects[i].skip_to_next = zero_skip;
            }
        }
    }

    /* Top inversion: switch t1 and t2 */
    if direction[1] < 0 {
        for i in 0..num_con {
            std::mem::swap(&mut connects[i].t1, &mut connects[i].t2);
        }
    }
}

/// Original static `outsideMeshLimits` (`mkmesh.c:1503`).
///
/// Tests for whether a triangle is entirely outside the limit.
pub fn outside_mesh_limits(p1: &Ipoint, p2: &Ipoint, p3: &Ipoint) -> i32 {
    let mesh_min = MESH_MIN.with(|c| c.get());
    let mesh_max = MESH_MAX.with(|c| c.get());
    if (p1.x < mesh_min.x || p1.x > mesh_max.x || p1.y < mesh_min.y || p1.y > mesh_max.y)
        && (p2.x < mesh_min.x || p2.x > mesh_max.x || p2.y < mesh_min.y || p2.y > mesh_max.y)
        && (p3.x < mesh_min.x || p3.x > mesh_max.x || p3.y < mesh_min.y || p3.y > mesh_max.y)
    {
        return 1;
    }
    0
}

/// Original static `chunkAddTriangle` (`mkmesh.c:1516`).
///
/// Adds one triangle to the mesh, exchanging second and third points if
/// `inside` is set.
pub fn chunk_add_triangle(
    mesh: &mut Imesh,
    i1: i32,
    i2: i32,
    i3: i32,
    maxsize: &mut i32,
    inside: i32,
) {
    let mut o2 = i2;
    let mut o3 = i3;
    if inside != 0 {
        o2 = i3;
        o3 = i2;
    }

    if mesh.list.is_empty() {
        chunk_mesh_add_index(mesh, IMOD_MESH_BGNPOLY, maxsize);
    }
    chunk_mesh_add_index(mesh, i1, maxsize);
    chunk_mesh_add_index(mesh, o2, maxsize);
    chunk_mesh_add_index(mesh, o3, maxsize);
}

/// Original: `CHUNKSIZE` (`mkmesh.c:1533`).
const CHUNKSIZE: i32 = 8192;

/// Original: `chunkMeshAddIndex` (`mkmesh.c:1540`).
///
/// Adds `index` to the index list of `mesh`, reserving new memory in large
/// chunks to avoid expensive frequent reallocations.  `maxlist` specifies the
/// current size of the allocated list and is returned with a new size when it
/// becomes larger.
pub fn chunk_mesh_add_index(mesh: &mut Imesh, index: i32, maxlist: &mut i32) -> i32 {
    if mesh.list.len() as i32 >= *maxlist {
        if !mesh.list.is_empty() {
            *maxlist += CHUNKSIZE;
        } else {
            *maxlist = CHUNKSIZE;
        }
        mesh.list.reserve(*maxlist as usize - mesh.list.len());
    }
    mesh.list.push(index);
    0
}

/// Original: `makeCapMesh` (`mkmesh.c:1562`).
///
/// Makes a cap mesh connecting a contour to a point in the given direction.
pub fn make_cap_mesh(
    cont: &Icont,
    cm: &Ipoint,
    meshdir: i32,
    props: &DrawProps,
    state: i32,
    state_test: i32,
) -> Option<Imesh> {
    let mut m = imod_mesh_new()?.remove(0);

    imod_mesh_add_vert(&mut m, cm);

    for pt in 0..cont.pts.len() {
        let npt = (pt + 1) % cont.pts.len();
        let point = cont.pts[pt];
        imod_mesh_add_vert(&mut m, &point);
        if outside_mesh_limits(cm, &cont.pts[pt], &cont.pts[npt]) != 0 {
            continue;
        }
        if state & state_test != 0 {
            let lsize = m.list.len() as i32;
            istore_generate_items(&mut m.store, props, state, lsize + 1, state_test);
            istore_generate_items(&mut m.store, props, state, lsize + 2, state_test);
            istore_generate_items(&mut m.store, props, state, lsize + 3, state_test);
        }
        imod_mesh_add_index(&mut m, IMOD_MESH_BGNPOLY);
        imod_mesh_add_index(&mut m, if meshdir != 0 { 0 } else { pt as i32 + 1 });
        imod_mesh_add_index(&mut m, npt as i32 + 1);
        imod_mesh_add_index(&mut m, if meshdir != 0 { pt as i32 + 1 } else { 0 });
        imod_mesh_add_index(&mut m, IMOD_MESH_ENDPOLY);
    }

    imod_mesh_add_index(&mut m, IMOD_MESH_END);
    Some(m)
}

/// Original: `makeTubeCont` (`mkmesh.c:1592`).
///
/// Makes a contour for a tube mesh.
pub fn make_tube_cont(
    cont: &mut Icont,
    loc: &Ipoint,
    n: &Ipoint,
    scale: &Ipoint,
    tube_diameter: f32,
    slices: i32,
) -> i32 {
    let mut spt = Ipoint::default();
    let mut tpt = Ipoint::default();
    let mut cpt = Ipoint::default();
    let Some(mut mat) = imod_mat_new(3) else {
        return 0;
    };
    let Some(mut rmat) = imod_mat_new(3) else {
        return 0;
    };
    let astep = 360.0 / slices as f64;
    let mut a: f64;
    let mut b: f64;

    let mut rscale = Ipoint::default();

    rscale.x = 1.0f32 / scale.x;
    rscale.y = 1.0f32 / scale.y;
    rscale.z = 1.0f32 / scale.z;

    b = (n.z as f64).acos();
    b *= 57.29578;

    /* DNM: modify method of getting rotation matrix to end up in correct
    quadrant, using atan2 */

    imod_mat_rot(&mut mat, b, B3D_Y);
    a = (n.y as f64).atan2(n.x as f64);
    a *= 57.29578;
    imod_mat_rot(&mut mat, a, B3D_Z);

    spt.x = 0.0f32;
    spt.y = tube_diameter * 0.5f32;
    spt.z = 0.0f32;

    imod_mat_scale(&mut mat, &rscale); /* DNM: Move this outside the loop */
    for _sl in 0..slices {
        imod_mat_rot(&mut rmat, astep, B3D_Z);
        imod_mat_transform(&rmat, &spt, &mut tpt);
        imod_mat_transform(&mat, &tpt, &mut cpt);

        cpt.x += loc.x;
        cpt.y += loc.y;
        cpt.z += loc.z;
        imod_point_append(cont, cpt);
    }
    imod_mat_delete(&mut mat);
    imod_mat_delete(&mut rmat);
    0
}

/// Original static `circle_top_and_direction` (`mkmesh.c:1641`).
///
/// Finds the point at which the back-transformed circle reaches its top, and
/// returns 0 for a ccw or 1 for a cw circle.
pub fn circle_top_and_direction(cont: &Icont, mat: &Imat, ptop: &mut i32) -> i32 {
    let mut reverse = 0;
    let mut ytop = 0.0f32;
    let mut xtop = 0.0f32;
    let mut tpt = Ipoint::default();

    /* Find pt at which Y reaches a maximum */
    for pt in 0..=cont.pts.len() as i32 {
        let mut pt2 = pt;
        if pt2 == cont.pts.len() as i32 {
            pt2 = 0;
        }
        imod_mat_transform(mat, &cont.pts[pt2 as usize], &mut tpt);
        if pt == 0 || tpt.y > ytop {
            ytop = tpt.y;
            xtop = tpt.x;
            *ptop = pt2;
        }
        /* if the LAST point was a maximum, determine whether there's
        a reversal based on change in X */
        if pt == *ptop + 1 {
            reverse = i32::from(tpt.x > xtop);
        }
    }
    reverse
}

/// Original: `joinTubeCont` (`mkmesh.c:1668`).
///
/// Joins two tube contours together.  Returns `None` for error.
pub fn join_tube_cont(
    c1: &Icont,
    c2: &Icont,
    norm: &Ipoint,
    props1: &mut DrawProps,
    state1_in: i32,
    props2: &mut DrawProps,
    state2_in: i32,
) -> Option<Imesh> {
    let mut state1 = state1_in;
    let mut state2 = state2_in;
    let mut pt1: i32 = 0;
    let mut pt2: i32 = 0;
    let mut mat = imod_mat_new(3)?;
    let mut mesh = imod_mesh_new()?.remove(0);
    let state_test = CHANGED_COLOR | CHANGED_FCOLOR | CHANGED_TRANS | CHANGED_VALUE1;
    let gen_items = i32::from(((state1 | state2) & state_test) != 0);

    let mpt = c1.pts.len() as i32;
    let mpt2 = c2.pts.len() as i32;
    let maxpt = mpt.max(mpt2);

    /* Get matrix that rotates this central normal to Z axis */
    let mut b = (norm.z as f64).acos();
    b *= 57.29578;
    let mut a = (norm.y as f64).atan2(norm.x as f64);
    a *= 57.29578;
    imod_mat_rot(&mut mat, -a, B3D_Z);
    imod_mat_rot(&mut mat, -b, B3D_Y);

    /* Start at top point of each back-transformed circle */

    let reverse1 = circle_top_and_direction(c1, &mat, &mut pt1);
    let reverse2 = circle_top_and_direction(c2, &mat, &mut pt2);

    /* and go backwards for second contour if reverse flag doesn't match */

    let idir2 = if reverse1 == reverse2 { 1 } else { -1 };

    /* If either trans change flag is set, set both and make sure trans state
    is the same (both 0 or both non-zero) */
    if (state1 | state2) & CHANGED_TRANS != 0 {
        if props1.trans != 0 && props2.trans == 0 {
            props2.trans = 1;
        }
        if props2.trans != 0 && props1.trans == 0 {
            props1.trans = 1;
        }
        state1 |= CHANGED_TRANS;
        state2 |= CHANGED_TRANS;
    }

    let vsize = (mpt + mpt2) as usize;
    let lsize = (3 * (mpt + mpt2 + 1)) as usize;
    mesh.vert = vec![Ipoint::default(); vsize];
    mesh.list = vec![0i32; lsize];

    /* Load the vertices first, in the right order for each circle */
    let mut pt: i32 = 0;
    while pt < mpt {
        mesh.vert[pt as usize] = c1.pts[pt1 as usize];
        pt1 += 1;
        if pt1 == mpt {
            pt1 = 0;
        }
        pt += 1;
    }

    while pt < mpt + mpt2 {
        mesh.vert[pt as usize] = c2.pts[pt2 as usize];
        pt2 += idir2;
        if pt2 == mpt2 {
            pt2 = 0;
        }
        if pt2 < 0 {
            pt2 = mpt2 - 1;
        }
        pt += 1;
    }

    let mut last1 = 0i32;
    let mut last2 = 0i32;
    let mut k = 0usize;
    mesh.list[k] = IMOD_MESH_BGNPOLY;
    k += 1;
    for pt in 0..maxpt {
        /* Get next point in c1, and the point it matches in c2 */
        let npt = (pt + 1) % maxpt;
        let next1 = ((mpt as f64 * npt as f64) / maxpt as f64 + 0.5).floor() as i32 % mpt;
        let next2 = ((mpt2 as f64 * npt as f64) / maxpt as f64 + 0.5).floor() as i32 % mpt2;

        /* add triangle with base in c1, and triangle with base in c2 only if the
        matches are different */
        if last1 != next1 {
            mesh.list[k] = mpt + last2;
            k += 1;
            mesh.list[k] = last1;
            k += 1;
            mesh.list[k] = next1;
            k += 1;
            if gen_items != 0 {
                istore_generate_items(&mut mesh.store, props2, state2, k as i32 - 3, state_test);
                istore_generate_items(&mut mesh.store, props1, state1, k as i32 - 2, state_test);
                istore_generate_items(&mut mesh.store, props1, state1, k as i32 - 1, state_test);
            }
        }

        if next2 != last2 {
            mesh.list[k] = mpt + last2;
            k += 1;
            mesh.list[k] = next1;
            k += 1;
            mesh.list[k] = mpt + next2;
            k += 1;
            if gen_items != 0 {
                istore_generate_items(&mut mesh.store, props2, state2, k as i32 - 3, state_test);
                istore_generate_items(&mut mesh.store, props1, state1, k as i32 - 2, state_test);
                istore_generate_items(&mut mesh.store, props2, state2, k as i32 - 1, state_test);
            }
        }

        last1 = next1;
        last2 = next2;
    }
    mesh.list[k] = IMOD_MESH_ENDPOLY;
    k += 1;
    mesh.list[k] = IMOD_MESH_END;

    imod_mat_delete(&mut mat);
    Some(mesh)
}

/// C `atoi` over a `&str`, which stops at the first non-numeric character.
fn atoi(value: &str) -> i32 {
    let trimmed = value.trim_start();
    let mut end = 0;
    let bytes = trimmed.as_bytes();
    if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
        end += 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    trimmed[..end].parse::<i32>().unwrap_or(0)
}
