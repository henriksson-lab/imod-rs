//! Translation of `IMOD/libmesh/remesh.c` -- remakes mesh with efficient
//! storage and adds normals.

use std::cell::Cell;

use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_error};
use crate::imod::libimod::imesh::{
    IMESH_FLAG_RES_SHIFT, IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2,
    IMOD_MESH_END, IMOD_MESH_ENDPOLY, imesh_resol, imod_mesh_add_index, imod_mesh_new,
    imod_mesh_poly_norm_factors, imod_meshes_delete, imodel_mesh_add,
};
use crate::imod::libimod::imodel::{Imesh, Ipoint};
use crate::imod::libimod::istore::istore_copy_cont_surf_items;
use crate::imod::libmesh::mkmesh::chunk_mesh_add_index;
use crate::imod::libmesh::skinobj::skin_report_time;

thread_local! {
    /// Original static `newPolyNorm` (`remesh.c:21`).
    static NEW_POLY_NORM: Cell<i32> = const { Cell::new(1) };
}

/// Original: `imeshSetNewPolyNorm` (`remesh.c:22`).
pub fn imesh_set_new_poly_norm(value: i32) {
    NEW_POLY_NORM.with(|c| c.set(value));
}

/// Original: `XDIV` (`remesh.c:27`).
const XDIV: i32 = 10;

/// Original: `imeshReMeshNormal` (`remesh.c:41`).
///
/// Remakes meshes into one mesh per surface and time to store data more
/// efficiently, and calculates normals for all vertices.  The array of meshes
/// is in `meshes`, and `size` specifies the number of meshes.  Only meshes
/// matching the resolution given by `resol` will be processed; others will be
/// copied to the output.  The normals are scaled by the values in `scale`.
/// THIS FUNCTION DELETES THE INPUT MESH BEFORE RETURNING.
/// Returns `None` for error.
pub fn imesh_remesh_normal(
    meshes: &mut Vec<Imesh>,
    size: &mut i32,
    scale: Option<&Ipoint>,
    resol: i32,
) -> Option<Vec<Imesh>> {
    let mut nm: Imesh;
    let mut remesh: Vec<Imesh> = Vec::new();
    let mut npt = Ipoint::default();
    let mut surfsize: i32 = 0;
    let mut timesize: i32 = 0;
    let msize: i32 = *size;
    let mut linc: i32 = 0;
    let mut vbase: i32 = 0;
    let mut n_add: i32 = 0;
    let (mut l1, mut l2): (i32, i32);
    let mut last_poly_norm: i32;
    let mut intz: i32;
    let mut vecx: f32;
    let mut ptmp = [0i32; 3];

    if msize == 0 {
        return None;
    }

    for m in 0..msize as usize {
        /* count only the meshes where the resolution value matches */
        if imesh_resol(meshes[m].flag) == resol {
            if meshes[m].surf as i32 > surfsize {
                surfsize = meshes[m].surf as i32;
            }
            if meshes[m].time as i32 > timesize {
                timesize = meshes[m].time as i32;
            }
            if !meshes[m].store.is_empty() {
                NEW_POLY_NORM.with(|c| c.set(1));
            }
        } else {
            /* otherwise, move the mesh to the output mesh */
            imodel_mesh_add(Some(&meshes[m]), &mut remesh);
            meshes[m].vert = Vec::new();
            meshes[m].list = Vec::new();
            meshes[m].store = Vec::new();
        }
    }
    surfsize += 1;
    timesize += 1;
    let new_poly_norm = NEW_POLY_NORM.with(|c| c.get());
    let out_code = if new_poly_norm != 0 {
        IMOD_MESH_BGNPOLYNORM2
    } else {
        IMOD_MESH_BGNPOLYNORM
    };

    skin_report_time("Starting ReMesh");

    for surf in 0..surfsize {
        for time in 0..timesize {
            nm = imod_mesh_new()?.remove(0);

            /* DNM 6/20/01: in the course of adding time support, had it transfer
            surface number to the mesh also */
            let mut maxlist: i32 = 0;
            nm.surf = surf as i16;
            nm.time = time as i16;

            /* Find min and max Z and X values */
            let mut zmax: i32 = -10000000;
            let mut zmin: i32 = 10000000;
            let mut xmax: f32 = -1.0e30;
            let mut xmin: f32 = 1.0e30;
            for m in 0..*size as usize {
                if meshes[m].surf as i32 == surf && meshes[m].time as i32 == time {
                    let mesh = &meshes[m];
                    let mut l: i32 = 0;
                    while l < mesh.list.len() as i32 {
                        linc = 1;
                        vbase = 0;
                        if mesh.list[l as usize] == IMOD_MESH_BGNPOLY
                            || imod_mesh_poly_norm_factors(
                                mesh.list[l as usize],
                                &mut linc,
                                &mut vbase,
                                &mut n_add,
                            ) != 0
                        {
                            l += 1;
                            while mesh.list[l as usize] != IMOD_MESH_ENDPOLY {
                                for _ivt in 0..3 {
                                    let iv = mesh.list[(l + vbase) as usize] as usize;
                                    intz = (mesh.vert[iv].z as f64).floor() as i32;
                                    if intz > zmax {
                                        zmax = intz;
                                    }
                                    if intz < zmin {
                                        zmin = intz;
                                    }
                                    vecx = mesh.vert[iv].x;
                                    if vecx > xmax {
                                        xmax = vecx;
                                    }
                                    if vecx < xmin {
                                        xmin = vecx;
                                    }
                                    l += linc;
                                }
                            }
                        }
                        l += 1;
                    }
                }
            }

            if zmax < zmin {
                continue;
            }

            let tablesize = (XDIV * (zmax + 1 - zmin)) as usize;
            let delx = ((xmax as f64 + 1.0 - xmin as f64) / XDIV as f64) as f32;
            /* Count the number of each Z and fractional X value */
            let mut numatz: Vec<i32> = vec![0; tablesize];
            let mut cumatz: Vec<i32> = vec![0; tablesize + 1];
            let mut vecatz: Vec<Vec<Ipoint>> = vec![Vec::new(); tablesize];

            for m in 0..*size as usize {
                if meshes[m].surf as i32 == surf && meshes[m].time as i32 == time {
                    let mesh = &meshes[m];
                    let mut l: i32 = 0;
                    while l < mesh.list.len() as i32 {
                        linc = 1;
                        vbase = 0;
                        if mesh.list[l as usize] == IMOD_MESH_BGNPOLY
                            || imod_mesh_poly_norm_factors(
                                mesh.list[l as usize],
                                &mut linc,
                                &mut vbase,
                                &mut n_add,
                            ) != 0
                        {
                            l += 1;
                            while mesh.list[l as usize] != IMOD_MESH_ENDPOLY {
                                for _ivt in 0..3 {
                                    let iv = mesh.list[(l + vbase) as usize] as usize;
                                    intz = (mesh.vert[iv].z as f64).floor() as i32;
                                    vecx = mesh.vert[iv].x;
                                    let indx = ((vecx - xmin) / delx) as i32;
                                    numatz[(indx + XDIV * (intz - zmin)) as usize] += 1;
                                    l += linc;
                                }
                            }
                        }
                        l += 1;
                    }
                }
            }

            /* allocate arrays for the points, rezero the counters */
            for l in 0..tablesize {
                if numatz[l] != 0 {
                    vecatz[l] = vec![Ipoint::default(); numatz[l] as usize];
                    numatz[l] = 0;
                }
            }

            /* Put points into arrays, eliminating duplicates as we go */
            for m in 0..*size as usize {
                let mesh = &meshes[m];
                if mesh.surf as i32 == surf && mesh.time as i32 == time && !mesh.vert.is_empty() {
                    let mut gotit: Vec<u8> = vec![0; mesh.vert.len()];

                    let mut l: i32 = 0;
                    while l < mesh.list.len() as i32 {
                        linc = 1;
                        vbase = 0;
                        if mesh.list[l as usize] == IMOD_MESH_BGNPOLY
                            || imod_mesh_poly_norm_factors(
                                mesh.list[l as usize],
                                &mut linc,
                                &mut vbase,
                                &mut n_add,
                            ) != 0
                        {
                            l += 1;
                            while mesh.list[l as usize] != IMOD_MESH_ENDPOLY {
                                for _ivt in 0..3 {
                                    let indvert = mesh.list[(l + vbase) as usize] as usize;
                                    if gotit[indvert] == 0 {
                                        gotit[indvert] = 1;
                                        intz = (mesh.vert[indvert].z as f64).floor() as i32;
                                        let vecx = mesh.vert[indvert].x;
                                        let indx = ((vecx - xmin) / delx) as i32;
                                        let intz = (indx + XDIV * (intz - zmin)) as usize;
                                        /* Take out this search and use qsort below */
                                        let n = numatz[intz] as usize;
                                        vecatz[intz][n] = mesh.vert[indvert];
                                        numatz[intz] += 1;
                                    }
                                    l += linc;
                                }
                            }
                        }
                        l += 1;
                    }
                }
            }

            /* Sort the vertices by X/Y, eliminate duplicates */
            for l in 0..tablesize {
                if numatz[l] != 0 {
                    let n = numatz[l] as usize;
                    let iptz = &mut vecatz[l];
                    iptz[..n].sort_by(ptcompare_ord);
                    l2 = 1;
                    l1 = 1;
                    while l1 < numatz[l] {
                        if ptcompare(&iptz[(l2 - 1) as usize], &iptz[l1 as usize]) != 0 {
                            iptz[l2 as usize] = iptz[l1 as usize];
                            l2 += 1;
                        }
                        l1 += 1;
                    }
                    numatz[l] = l2;
                }
            }

            cumatz[0] = 0;
            for l in 0..tablesize {
                cumatz[l + 1] = cumatz[l] + 2 * numatz[l];
            }

            /* copy points to mesh data, zero out normal space also */
            nm.vert = vec![Ipoint::default(); cumatz[tablesize] as usize];
            {
                let mut ind = 0usize;
                for l in 0..tablesize {
                    for pt in 0..numatz[l] as usize {
                        nm.vert[ind] = vecatz[l][pt];
                        ind += 1;
                        nm.vert[ind].x = 0.;
                        nm.vert[ind].y = 0.;
                        nm.vert[ind].z = 0.;
                        ind += 1;
                    }
                }
            }

            skin_report_time("Built tables");

            /* add normals to mesh */
            for m in 0..msize as usize {
                if meshes[m].surf as i32 != surf || meshes[m].time as i32 != time {
                    continue;
                }
                if meshes[m].list.is_empty() {
                    continue;
                }
                chunk_mesh_add_index(&mut nm, out_code, &mut maxlist);
                last_poly_norm = -1;
                let mut l: i32 = 0;
                while l < meshes[m].list.len() as i32 {
                    linc = 1;
                    vbase = 0;
                    if meshes[m].list[l as usize] == IMOD_MESH_BGNPOLY
                        || imod_mesh_poly_norm_factors(
                            meshes[m].list[l as usize],
                            &mut linc,
                            &mut vbase,
                            &mut n_add,
                        ) != 0
                    {
                        /* Preserve polynorms polygons: start a new polygon if this is a
                        polynorm or if the last one was */
                        if (last_poly_norm >= 0
                            && imod_mesh_poly_norm_factors(
                                meshes[m].list[l as usize],
                                &mut linc,
                                &mut vbase,
                                &mut n_add,
                            ) != 0)
                            || last_poly_norm > 0
                        {
                            chunk_mesh_add_index(&mut nm, IMOD_MESH_ENDPOLY, &mut maxlist);
                            chunk_mesh_add_index(&mut nm, out_code, &mut maxlist);
                        }
                        last_poly_norm = imod_mesh_poly_norm_factors(
                            meshes[m].list[l as usize],
                            &mut linc,
                            &mut vbase,
                            &mut n_add,
                        );
                        l += 1;
                        while meshes[m].list[l as usize] != IMOD_MESH_ENDPOLY {
                            for ivt in 0..3usize {
                                let indvert = meshes[m].list[(l + vbase) as usize] as usize;
                                let intz = (meshes[m].vert[indvert].z as f64).floor() as i32;
                                let vecx = meshes[m].vert[indvert].x;
                                let indx = ((vecx - xmin) / delx) as i32;
                                let intz = (indx + XDIV * (intz - zmin)) as usize;

                                let key = meshes[m].vert[indvert];
                                let found = bsearch_pairs(
                                    &key,
                                    &nm.vert,
                                    cumatz[intz] as usize,
                                    numatz[intz] as usize,
                                );
                                let pt = match found {
                                    Some(ind) => ind as i32,
                                    None => {
                                        b3d_error(
                                            Some(&mut ImodFile::Stderr),
                                            format_args!("Failed to find vertex in reduced list\n"),
                                        );
                                        cumatz[intz]
                                    }
                                };
                                ptmp[ivt] = pt;
                                l += linc;
                            }

                            /* Save only if it is truly a triangle */
                            if ptmp[1] != ptmp[2] && ptmp[0] != ptmp[2] && ptmp[0] != ptmp[1] {
                                for ivt in 0..3usize {
                                    if new_poly_norm == 0 {
                                        chunk_mesh_add_index(&mut nm, ptmp[ivt] + 1, &mut maxlist);
                                    }
                                    chunk_mesh_add_index(&mut nm, ptmp[ivt], &mut maxlist);
                                    let lsize = nm.list.len() as i32;
                                    istore_copy_cont_surf_items(
                                        &meshes[m].store,
                                        &mut nm.store,
                                        l + ivt as i32 - 3,
                                        lsize - 1,
                                        0,
                                    );
                                }
                            }
                        }
                    }
                    l += 1;
                }
                chunk_mesh_add_index(&mut nm, IMOD_MESH_ENDPOLY, &mut maxlist);
            }

            imod_mesh_add_index(&mut nm, IMOD_MESH_END);

            skin_report_time("Found Indexes");

            /* calculate normals for all points. */
            {
                let mut l: i32 = 0;
                while l < nm.list.len() as i32 {
                    if imod_mesh_poly_norm_factors(
                        nm.list[l as usize],
                        &mut linc,
                        &mut vbase,
                        &mut n_add,
                    ) != 0
                    {
                        l += 1;
                        while nm.list[l as usize] != IMOD_MESH_ENDPOLY {
                            let p1 = nm.vert[nm.list[(l + vbase + 2 * linc) as usize] as usize];
                            let p2 = nm.vert[nm.list[(l + vbase) as usize] as usize];
                            let p3 = nm.vert[nm.list[(l + vbase + linc) as usize] as usize];
                            imesh_normal(&mut npt, &p1, &p2, &p3, scale);
                            for _ivt in 0..3 {
                                let indvert = (nm.list[l as usize] + n_add) as usize;
                                nm.vert[indvert].x += npt.x;
                                nm.vert[indvert].y += npt.y;
                                nm.vert[indvert].z += npt.z;
                                l += linc;
                            }
                        }
                    }
                    l += 1;
                }
            }

            nm.flag |= (resol as u32) << IMESH_FLAG_RES_SHIFT;
            if !nm.vert.is_empty() && !nm.list.is_empty() {
                imodel_mesh_add(Some(&nm), &mut remesh);
            }
        }
    }
    *size = remesh.len() as i32;

    imod_meshes_delete(Some(std::mem::take(meshes)), msize);
    skin_report_time("Calculated Normals");

    Some(remesh)
}

/// Original: `imeshNormal` (`remesh.c:414`).
///
/// Computes a normalized normal in `n` from the three points `p1`, `p2`, and
/// `p3`.  Scales the points by the scaling in `sp` first if `sp` is not None.
pub fn imesh_normal(n: &mut Ipoint, p1: &Ipoint, p2: &Ipoint, p3: &Ipoint, sp: Option<&Ipoint>) {
    let mut dist: f64;
    let sdist: f64;
    let mut v1 = Ipoint::default();
    let mut v2 = Ipoint::default();

    v1.x = p3.x - p2.x;
    v1.y = p3.y - p2.y;
    v1.z = p3.z - p2.z;

    v2.x = p1.x - p2.x;
    v2.y = p1.y - p2.y;
    v2.z = p1.z - p2.z;
    if let Some(sp) = sp {
        v1.x *= sp.x;
        v1.y *= sp.y;
        v1.z *= sp.z;
        v2.x *= sp.x;
        v2.y *= sp.y;
        v2.z *= sp.z;
    }

    n.x = (v1.y * v2.z) - (v1.z * v2.y);
    n.y = (v1.z * v2.x) - (v1.x * v2.z);
    n.z = (v1.x * v2.y) - (v1.y * v2.x);

    /* now normalize n ; x^2 + y^2 + z^2 = 1 */

    sdist = ((n.x * n.x) + (n.y * n.y) + (n.z * n.z)) as f64;
    dist = sdist.sqrt();
    if dist == 0.0 {
        n.x = 0.;
        n.y = 0.;
        n.z = -1.;
    } else {
        dist = 1. / dist;
        n.x = (n.x as f64 * dist) as f32;
        n.y = (n.y as f64 * dist) as f32;
        n.z = (n.z as f64 * dist) as f32;
    }
}

/// Original static `ptcompare` (`remesh.c:459`).
pub fn ptcompare(pt1: &Ipoint, pt2: &Ipoint) -> i32 {
    if pt1.x < pt2.x {
        return -1;
    }
    if pt1.x > pt2.x {
        return 1;
    }
    if pt1.y == pt2.y {
        return 0;
    }
    if pt1.y > pt2.y {
        return 1;
    }
    -1
}

/// `ptcompare` as an ordering, for the source's `qsort` call.
fn ptcompare_ord(pt1: &Ipoint, pt2: &Ipoint) -> std::cmp::Ordering {
    ptcompare(pt1, pt2).cmp(&0)
}

/// The source's `bsearch` over `nm->vert` with a stride of two `Ipoint`s,
/// reproducing glibc's loop.  Returns the index into `vert` of the match.
fn bsearch_pairs(key: &Ipoint, vert: &[Ipoint], base: usize, nmemb: usize) -> Option<usize> {
    let mut l = 0usize;
    let mut u = nmemb;
    while l < u {
        let idx = (l + u) / 2;
        let p = base + 2 * idx;
        let comparison = ptcompare(key, &vert[p]);
        match comparison.cmp(&0) {
            std::cmp::Ordering::Less => u = idx,
            std::cmp::Ordering::Greater => l = idx + 1,
            std::cmp::Ordering::Equal => return Some(p),
        }
    }
    None
}
