//! Library of mesh functions, from `IMOD/libimod/imesh.c` and the constants,
//! macros and `MeshParams` declaration of `IMOD/include/imesh.h` and
//! `IMOD/include/imodel.h:335`.
//!
//! The translated `Imesh` in `imodel.rs` keeps the malloc'd `vert`/`list`
//! arrays and their `vsize`/`lsize` counts as `Vec`s, and has no `vertBuf`
//! run-time member (that is 3dmod's OpenGL vertex buffer, which this
//! translation never allocates), so the `mesh[sh].vertBuf = NULL` of
//! `imodMeshesNew` and the `newMesh->vertBuf = NULL` of `imodMeshDup` have no
//! counterpart here.
#![allow(dead_code)]

use super::imodel::{Icont, Imesh, Ipoint};
use super::ipoint::{imod_point_append, imod_point_cross, imod_point_normalize};

/// Original: `IMOD_MESH_BGNTRI` (`imesh.h:19`).
pub const IMOD_MESH_BGNTRI: i32 = -11;
/// Original: `IMOD_MESH_ENDTRI` (`imesh.h:20`).
pub const IMOD_MESH_ENDTRI: i32 = -12;
/// Original: `IMOD_MESH_SWAP` (`imesh.h:21`).
pub const IMOD_MESH_SWAP: i32 = -10;
/// Original: `IMOD_MESH_END` (`imesh.h:26`).
pub const IMOD_MESH_END: i32 = -1;
/// Original: `IMOD_MESH_ENDPOLY` (`imesh.h:31`).
pub const IMOD_MESH_ENDPOLY: i32 = -22;
/// Original: `IMOD_MESH_NORMAL` (`imesh.h:33`).
pub const IMOD_MESH_NORMAL: i32 = -20;
/// Original: `IMOD_MESH_BGNPOLY` (`imesh.h:34`).
pub const IMOD_MESH_BGNPOLY: i32 = -21;
/// Original: `IMOD_MESH_BGNBIGPOLY` (`imesh.h:35`).
pub const IMOD_MESH_BGNBIGPOLY: i32 = -24;
/// Original: `IMOD_MESH_BGNPOLYNORM` (`imesh.h:42`).
pub const IMOD_MESH_BGNPOLYNORM: i32 = -23;
/// Original: `IMOD_MESH_BGNPOLYNORM2` (`imesh.h:43`).
pub const IMOD_MESH_BGNPOLYNORM2: i32 = -25;

/// Original: `IMESH_FLAG_NMAG` (`imesh.h:47`).
pub const IMESH_FLAG_NMAG: u32 = 1 << 16;
/// Original: `IMESH_FLAG_RES_SHIFT` (`imesh.h:48`).
pub const IMESH_FLAG_RES_SHIFT: u32 = 20;
/// Original: `IMESH_FLAG_RES_BITS` (`imesh.h:49`).
pub const IMESH_FLAG_RES_BITS: u32 = 15 << IMESH_FLAG_RES_SHIFT;
/// Original: `IMESH_THICKNESS_SHIFT` (`imesh.h:50`).
pub const IMESH_THICKNESS_SHIFT: u32 = 24;
/// Original: `IMESH_THICKNESS_BITS` (`imesh.h:51`).
pub const IMESH_THICKNESS_BITS: u32 = 63 << IMESH_THICKNESS_SHIFT;

/// Original: `imeshResol` (`imesh.h:53`).
pub fn imesh_resol(flag: u32) -> i32 {
    ((flag & IMESH_FLAG_RES_BITS) >> IMESH_FLAG_RES_SHIFT) as i32
}

/// Original: `imeshThickness` (`imesh.h:54`).
pub fn imesh_thickness(flag: u32) -> i32 {
    ((flag & IMESH_THICKNESS_BITS) >> IMESH_THICKNESS_SHIFT) as i32
}

/// Original: `DEFAULT_VALUE` (`imesh.h:56`).
pub const DEFAULT_VALUE: i32 = 0x7fffffff;
/// Original: `DEFAULT_FLOAT` (`imesh.h:57`).
pub const DEFAULT_FLOAT: f32 = 1.0e30;

/// Original: `IMESH_MK_FAST` (`mkmesh.h:15`).
pub const IMESH_MK_FAST: u32 = 1 << 2;
/// Original: `IMESH_MK_SKIP` (`mkmesh.h:16`).
pub const IMESH_MK_SKIP: u32 = 1 << 3;
/// Original: `IMESH_MK_NORM` (`mkmesh.h:17`).
pub const IMESH_MK_NORM: u32 = 1 << 4;
/// Original: `IMESH_MK_STRAY` (`mkmesh.h:18`).
pub const IMESH_MK_STRAY: u32 = 1 << 5;
/// Original: `IMESH_MK_SURF` (`mkmesh.h:19`).
pub const IMESH_MK_SURF: u32 = 1 << 6;
/// Original: `IMESH_MK_TUBE` (`mkmesh.h:20`).
pub const IMESH_MK_TUBE: u32 = 1 << 7;
/// Original: `IMESH_MK_TIME` (`mkmesh.h:21`).
pub const IMESH_MK_TIME: u32 = 1 << 8;
/// Original: `IMESH_MK_CAP_TUBE` (`mkmesh.h:22`).
pub const IMESH_MK_CAP_TUBE: u32 = 1 << 9;
/// Original: `IMESH_MK_IS_COPY` (`mkmesh.h:23`).
pub const IMESH_MK_IS_COPY: u32 = 1 << 10;
/// Original: `IMESH_MK_USE_MEAN` (`mkmesh.h:24`).
pub const IMESH_MK_USE_MEAN: u32 = 1 << 11;
/// Original: `IMESH_MK_NO_WARN` (`mkmesh.h:25`).
pub const IMESH_MK_NO_WARN: u32 = 1 << 12;
/// Original: `IMESH_MK_CAP_DOME` (`mkmesh.h:26`).
pub const IMESH_MK_CAP_DOME: u32 = 1 << 13;
/// Original: `IMESH_CAP_OFF` (`mkmesh.h:28`).
pub const IMESH_CAP_OFF: i32 = 0;
/// Original: `IMESH_CAP_END` (`mkmesh.h:29`).
pub const IMESH_CAP_END: i32 = 1;
/// Original: `IMESH_CAP_ALL` (`mkmesh.h:30`).
pub const IMESH_CAP_ALL: i32 = 2;

/// Original: `MeshParams` / `struct Meshing_Param` (`imodel.h:335`).
#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct MeshParams {
    pub flags: u32,
    pub cap: i32,
    pub passes: i32,
    pub cap_skip_nz: i32,
    pub incz_low_res: i32,
    pub incz_high_res: i32,
    pub minz: i32,
    pub maxz: i32,
    pub spare_int: i32,
    pub overlap: f32,
    pub tube_diameter: f32,
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
    pub tol_low_res: f32,
    pub tol_high_res: f32,
    pub flat_crit: f32,
    pub spare_float: f32,
    pub cap_skip_zlist: Option<Vec<i32>>,
}

impl Default for MeshParams {
    /// The C structure is `malloc`ed without initialization; every caller runs
    /// `imeshParamsDefault` on it first, which this reproduces.
    fn default() -> Self {
        let mut params = MeshParams {
            flags: 0,
            cap: 0,
            passes: 0,
            cap_skip_nz: 0,
            incz_low_res: 0,
            incz_high_res: 0,
            minz: 0,
            maxz: 0,
            spare_int: 0,
            overlap: 0.,
            tube_diameter: 0.,
            xmin: 0.,
            xmax: 0.,
            ymin: 0.,
            ymax: 0.,
            tol_low_res: 0.,
            tol_high_res: 0.,
            flat_crit: 0.,
            spare_float: 0.,
            cap_skip_zlist: None,
        };
        imesh_params_default(&mut params);
        params
    }
}

/// Original: `imodMeshNew` (`imesh.c:25`).
pub fn imod_mesh_new() -> Option<Vec<Imesh>> {
    imod_meshes_new(1)
}

/// Original: `imodMeshesNew` (`imesh.c:34`).
pub fn imod_meshes_new(size: i32) -> Option<Vec<Imesh>> {
    if size <= 0 {
        return None;
    }

    let mut mesh: Vec<Imesh> = Vec::with_capacity(size as usize);

    for _sh in 0..size {
        mesh.push(Imesh {
            vert: Vec::new(),
            list: Vec::new(),
            flag: 0,
            time: 0,
            surf: 0,
            store: Vec::new(),
        });
    }
    Some(mesh)
}

/// Original: `imodMeshGetIndex` (`imesh.c:62`).
pub fn imod_mesh_get_index(mesh: Option<&Imesh>, index: i32) -> i32 {
    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return IMOD_MESH_END,
    };
    if index < 0 || index >= mesh.list.len() as i32 {
        return IMOD_MESH_END;
    }
    mesh.list[index as usize]
}

/// Original: `imodMeshGetMaxIndex` (`imesh.c:71`).
pub fn imod_mesh_get_max_index(mesh: Option<&Imesh>) -> i32 {
    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return 0,
    };
    mesh.list.len() as i32
}

/// Original: `imodMeshGetMaxVert` (`imesh.c:78`).
pub fn imod_mesh_get_max_vert(mesh: Option<&Imesh>) -> i32 {
    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return 0,
    };
    mesh.vert.len() as i32
}

/// Original: `imodMeshGetVert` (`imesh.c:86`).
pub fn imod_mesh_get_vert(mesh: Option<&Imesh>, index: i32) -> Option<&Ipoint> {
    let mesh = mesh?;
    if index < 0 || index >= mesh.vert.len() as i32 {
        return None;
    }
    Some(&mesh.vert[index as usize])
}

/// Original: `imodMeshGetVerts` (`imesh.c:95`).
pub fn imod_mesh_get_verts(mesh: Option<&Imesh>) -> Option<&[Ipoint]> {
    let mesh = mesh?;
    if mesh.vert.is_empty() {
        return None;
    }
    Some(&mesh.vert)
}

/// Original: `imodMeshCopy` (`imesh.c:107`).
///
/// The C `memcpy(to, from, sizeof(Imesh))` transfers the two allocated arrays
/// by pointer, leaving both structures aliasing them; here they are cloned,
/// which is observationally the same in every caller because the source
/// mesh is either abandoned or freed with `free()` rather than
/// `imodMeshFreeData` afterwards.
pub fn imod_mesh_copy(from: Option<&Imesh>, to: Option<&mut Imesh>) -> i32 {
    let from = match from {
        Some(from) => from,
        None => return -1,
    };
    let to = match to {
        Some(to) => to,
        None => return -1,
    };
    *to = from.clone();
    0
}

/// Original: `imodMeshDup` (`imesh.c:118`).
pub fn imod_mesh_dup(mesh: Option<&Imesh>) -> Option<Imesh> {
    let mut new_mesh = imod_mesh_new()?.remove(0);
    imod_mesh_copy(mesh, Some(&mut new_mesh));
    let mesh = mesh?;
    new_mesh.vert = mesh.vert.clone();
    new_mesh.list = mesh.list.clone();
    new_mesh.store = mesh.store.clone();
    Some(new_mesh)
}

/// Original: `imodMeshDelete` (`imesh.c:141`).
pub fn imod_mesh_delete(mesh: Option<Vec<Imesh>>) -> i32 {
    imod_meshes_delete(mesh, 1)
}

/// Original: `imodMeshFreeData` (`imesh.c:149`).
pub fn imod_mesh_free_data(mesh: Option<&mut Imesh>) -> i32 {
    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return -1,
    };
    mesh.vert = Vec::new();
    mesh.list = Vec::new();
    mesh.store = Vec::new();
    0
}

/// Original: `imodMeshesDelete` (`imesh.c:163`).
pub fn imod_meshes_delete(mesh: Option<Vec<Imesh>>, size: i32) -> i32 {
    let mut mesh = match mesh {
        Some(mesh) => mesh,
        None => return -1,
    };
    for ms in 0..size as usize {
        if ms < mesh.len() {
            let entry = &mut mesh[ms];
            imod_mesh_free_data(Some(entry));
        }
    }
    drop(mesh);
    0
}

/// Original: `imodMeshAddIndex` (`imesh.c:179`).
pub fn imod_mesh_add_index(mesh: &mut Imesh, index: i32) -> i32 {
    mesh.list.push(index);
    0
}

/* Unused 7/4/05 */
/// Original: `imodMeshDeleteIndex` (`imesh.c:201`).
pub fn imod_mesh_delete_index(mesh: Option<&mut Imesh>, index: i32) {
    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return,
    };
    if index < 0 || index >= mesh.list.len() as i32 {
        return;
    }

    let lsize = mesh.list.len();
    for i in (index as usize + 1)..lsize {
        mesh.list[i - 1] = mesh.list[i];
    }
    mesh.list.truncate(lsize - 1);
}

/// Original: `imodMeshAddVert` (`imesh.c:217`).
pub fn imod_mesh_add_vert(mesh: &mut Imesh, vert: &Ipoint) -> i32 {
    mesh.vert.push(Ipoint {
        x: vert.x,
        y: vert.y,
        z: vert.z,
    });
    0
}

/* Unused 7/4/05 */
/// Original: `imodMeshAddNormal` (`imesh.c:238`).
pub fn imod_mesh_add_normal(mesh: Option<&mut Imesh>, normal: Option<&Ipoint>) -> i32 {
    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return -1,
    };
    let normal = match normal {
        Some(normal) => normal,
        None => return -1,
    };
    imod_mesh_add_vert(mesh, normal);
    imod_mesh_add_index(mesh, IMOD_MESH_NORMAL);
    let vsize = mesh.vert.len() as i32;
    imod_mesh_add_index(mesh, vsize - 1);
    0
}

/* Unused 7/4/05 */
/// Original: `imodMeshInsertIndex` (`imesh.c:256`).
pub fn imod_mesh_insert_index(mesh: &mut Imesh, val: i32, place: i32) {
    let mut l: i32;
    imod_mesh_add_index(mesh, val);

    l = mesh.list.len() as i32 - 1;
    while l > place {
        mesh.list[l as usize] = mesh.list[l as usize - 1];
        l -= 1;
    }
    mesh.list[l as usize] = val;
}

/// Original: `imodMeshNearestRes` (`imesh.c:275`).
pub fn imod_mesh_nearest_res(mesh: &[Imesh], size: i32, inres: i32, outres: &mut i32) -> i32 {
    let mut res: i32;
    let mut oldiff: i32;
    let mut newdiff: i32;
    let mut ndiff: i32 = 0;
    *outres = inres;
    if size == 0 || mesh.is_empty() {
        return -1;
    }
    *outres = imesh_resol(mesh[0].flag);

    for m in 1..size as usize {
        res = imesh_resol(mesh[m].flag);
        if res != *outres {
            ndiff = 1;
            oldiff = *outres - inres;
            if oldiff < 0 {
                oldiff = -oldiff;
            }
            newdiff = res - inres;
            if newdiff < 0 {
                newdiff = -newdiff;
            }
            /* take the resolution with the smallest difference from the
            desired one; or if there is a tie, take the one that is
            a higher number than desired rather than lower */
            if newdiff < oldiff || (newdiff > 0 && newdiff == oldiff) {
                *outres = res;
            }
        }
    }
    ndiff
}

/// Original: `imodMeshPolyNormFactors` (`imesh.c:312`).
pub fn imod_mesh_poly_norm_factors(
    start_code: i32,
    list_inc: &mut i32,
    vert_base: &mut i32,
    norm_add: &mut i32,
) -> i32 {
    if start_code != IMOD_MESH_BGNPOLYNORM && start_code != IMOD_MESH_BGNPOLYNORM2 {
        return 0;
    }
    if start_code == IMOD_MESH_BGNPOLYNORM {
        *list_inc = 2;
        *vert_base = 1;
        *norm_add = 0;
    } else {
        *list_inc = 1;
        *vert_base = 0;
        *norm_add = 1;
    }
    1
}

/* 2/25/06: deleted imodel_mesh_addlist and imodel_mesh_addvert which were
duplicates of imodMesh calls */

/// Original: `imodel_mesh_add` (`imesh.c:340`).
///
/// The C reallocates `mray` and returns the new array; here the array is the
/// `Vec` passed in and the `size` in/out argument is its length.
pub fn imodel_mesh_add(nmesh: Option<&Imesh>, mray: &mut Vec<Imesh>) -> i32 {
    let nmesh = match nmesh {
        Some(nmesh) => nmesh,
        None => return 0,
    };

    mray.push(Imesh::default());
    let size = mray.len() - 1;
    imod_mesh_copy(Some(nmesh), Some(&mut mray[size]));
    0
}

/// Original: `imodMeshRemovePairs` (`imesh.c:368`).
pub fn imod_mesh_remove_pairs(
    meshes: &mut Vec<Imesh>,
    retain_thick: i32,
    cleanup_vbd: Option<fn(&mut Imesh)>,
) -> i32 {
    let mut thick: i32;
    let mut new_size: usize = 0;
    let mut num_match: i32 = 0;

    // Loop through the mesh array, copying down ones with no thickness or the retained
    // thickness, and cleaning up data in the rest
    let size = meshes.len();
    for me in 0..size {
        thick = imesh_thickness(meshes[me].flag);
        if thick == 0 || thick == retain_thick {
            if new_size < me {
                let copy = meshes[me].clone();
                meshes[new_size] = copy;
            }
            new_size += 1;
            if thick == retain_thick {
                num_match += 1;
            }
        } else {
            if let Some(cleanup) = cleanup_vbd {
                cleanup(&mut meshes[me]);
            }
            imod_mesh_free_data(Some(&mut meshes[me]));
        }
    }
    meshes.truncate(new_size);
    num_match
}

/// Original: `imodMeshMakePairs` (`imesh.c:400`).
pub fn imod_mesh_make_pairs(
    meshes: &mut Vec<Imesh>,
    thickness: i32,
    cleanup_vbd: Option<fn(&mut Imesh)>,
) -> bool {
    let mut ind: i32;
    let mut new_size: usize;
    let mut list_inc: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;
    let half_thick: f32 = thickness as f32 / 2.;
    let mut pnt: Ipoint;
    let mut norm = Ipoint::default();
    if meshes.is_empty() || thickness == 0 {
        return false;
    }

    // Remove non-matching pairs; if the number of matching ones is two thirds of the size
    // then we have it already
    let num_kept = imod_mesh_remove_pairs(meshes, thickness, cleanup_vbd);
    if 3 * num_kept == meshes.len() as i32 * 2 {
        return true;
    }

    // Otherwise, something is wrong, get rid of them completely
    if num_kept != 0 {
        imod_mesh_remove_pairs(meshes, 0, cleanup_vbd);
    }
    if 3 * meshes.len() == 0 {
        return true;
    }

    let size = meshes.len();
    new_size = size;
    for me in 0..size {
        /* Make a pair of meshes that are duplicates */
        let mut dir: i32 = -1;
        while dir <= 1 {
            let mut new_mesh = match imod_mesh_dup(Some(&meshes[me])) {
                Some(new_mesh) => new_mesh,
                None => {
                    meshes.truncate(new_size);
                    return false;
                }
            };
            new_mesh.flag |= (thickness as u32) << IMESH_THICKNESS_SHIFT;
            meshes.push(Imesh::default());
            imod_mesh_copy(Some(&new_mesh), Some(&mut meshes[new_size]));
            new_size += 1;
            dir += 2;
        }

        /* Make the two offset surfaces by adding and subtracting the normal times half the
        thickness and inverting the normal for the inner surface */
        ind = 0;
        while ind < meshes[me].vert.len() as i32 {
            let i0 = ind as usize;
            let i1 = ind as usize + 1;
            norm.x = meshes[new_size - 2].vert[i1].x;
            norm.y = meshes[new_size - 2].vert[i1].y;
            norm.z = meshes[new_size - 2].vert[i1].z;
            meshes[new_size - 2].vert[i1].x = -meshes[new_size - 2].vert[i1].x;
            meshes[new_size - 2].vert[i1].y = -meshes[new_size - 2].vert[i1].y;
            meshes[new_size - 2].vert[i1].z = -meshes[new_size - 2].vert[i1].z;
            imod_point_normalize(&mut norm);
            pnt = Ipoint {
                x: meshes[new_size - 2].vert[i0].x,
                y: meshes[new_size - 2].vert[i0].y,
                z: meshes[new_size - 2].vert[i0].z,
            };
            meshes[new_size - 2].vert[i0].x = pnt.x - half_thick * norm.x;
            meshes[new_size - 2].vert[i0].y = pnt.y - half_thick * norm.y;
            meshes[new_size - 2].vert[i0].z = pnt.z - half_thick * norm.z;
            meshes[new_size - 1].vert[i0].x = pnt.x + half_thick * norm.x;
            meshes[new_size - 1].vert[i0].y = pnt.y + half_thick * norm.y;
            meshes[new_size - 1].vert[i0].z = pnt.z + half_thick * norm.z;
            ind += 2;
        }

        /* Also need to invert the order of triangles in the inner surface to keep that
        consistent with the normal direction */
        let mut i: i32 = 0;
        while i < meshes[new_size - 2].list.len() as i32 {
            let code = meshes[new_size - 2].list[i as usize];
            if code == IMOD_MESH_BGNBIGPOLY || code == IMOD_MESH_BGNPOLY {
                while meshes[new_size - 2].list[i as usize] != IMOD_MESH_ENDPOLY {
                    i += 1;
                }
            } else if code == IMOD_MESH_BGNPOLYNORM2 || code == IMOD_MESH_BGNPOLYNORM {
                let start = meshes[new_size - 2].list[i as usize];
                i += 1;
                imod_mesh_poly_norm_factors(start, &mut list_inc, &mut vert_base, &mut norm_add);
                while meshes[new_size - 2].list[i as usize] != IMOD_MESH_ENDPOLY {
                    ind = meshes[new_size - 2].list[(i + vert_base) as usize];
                    meshes[new_size - 2].list[(i + vert_base) as usize] =
                        meshes[new_size - 2].list[(i + vert_base + list_inc) as usize];
                    meshes[new_size - 2].list[(i + vert_base + list_inc) as usize] = ind;
                    i += 3 * list_inc;
                }
            }
            i += 1;
        }
    }
    true
}

/// Original: `imodMeshInterpCont` (`imesh.c:507`).
pub fn imod_mesh_interp_cont(
    listp: &[i32],
    vertp: &[Ipoint],
    ntriang: i32,
    firstv: i32,
    list_inc: i32,
    zadd: i32,
    cont: &mut Icont,
) {
    let mut ind1: i32;
    let mut ind2: i32;
    let mut jnd1: i32 = 0;
    let mut jnd2: i32 = 0;
    let mut jnd3: i32 = 0;
    let mut indv: i32;
    let mut jndv: i32;
    let mut done: i32;
    let mut z1: f32; /* z values of two candidate vertices */
    let mut z2: f32;
    let mut frac: f32; /* interpolation fraction */
    let mut jtri: i32; /* triangle indexes */
    let mut ptadd: Ipoint; /* Point to add */

    /* loop on triangles */
    for itri in 0..ntriang {
        indv = firstv + itri * 3 * list_inc;

        /* Look at the three pairs of vertices in
        triangle, see if any bracket zadd */
        for j in 0..3 {
            ind1 = listp[(indv + j * list_inc) as usize];
            ind2 = listp[(indv + ((j + 1) % 3) * list_inc) as usize];
            z1 = vertp[ind1 as usize].z;
            z2 = vertp[ind2 as usize].z;
            if !((z1 > zadd as f32 && z2 < zadd as f32) || (z1 < zadd as f32 && z2 > zadd as f32)) {
                continue;
            }

            /* If it brackets, look back to see that
            this pair of vertices hasn't been done
            already */
            done = 0;
            jtri = itri - 1;
            while jtri >= 0 {
                jndv = firstv + jtri * 3 * list_inc;
                jnd1 = listp[jndv as usize];
                jnd2 = listp[(jndv + list_inc) as usize];
                jnd3 = listp[(jndv + 2 * list_inc) as usize];
                if (ind1 == jnd1 && ind2 == jnd2)
                    || (ind2 == jnd1 && ind1 == jnd2)
                    || (ind1 == jnd2 && ind2 == jnd3)
                    || (ind2 == jnd2 && ind1 == jnd3)
                    || (ind1 == jnd3 && ind2 == jnd1)
                    || (ind2 == jnd3 && ind1 == jnd1)
                {
                    done = 1;
                    break;
                }
                jtri -= 1;
            }

            /* If there is a duplicate, see if this is
            the second triangle, and if it is the
            first point of first triangle that
            matches - then need to swap the points */
            if done != 0 {
                if itri == 1
                    && ((ind1 == jnd1 && ind2 == jnd2)
                        || (ind2 == jnd1 && ind1 == jnd2)
                        || (vertp[jnd1 as usize].z == vertp[jnd2 as usize].z
                            && ((ind1 == jnd2 && ind2 == jnd3) || (ind2 == jnd2 && ind1 == jnd3))))
                {
                    ptadd = cont.pts[0];
                    cont.pts[0] = cont.pts[1];
                    cont.pts[1] = ptadd;
                }

                continue;
            }

            /* It passes the test, add interpolated
            point */
            frac = (zadd as f32 - z1) / (z2 - z1);
            /* `(1. - frac)` is a C double because `1.` is; `frac * vertp[].x`
            stays float because both operands are.  Both widths are kept. */
            ptadd = Ipoint {
                x: ((1. - frac as f64) * vertp[ind1 as usize].x as f64
                    + (frac * vertp[ind2 as usize].x) as f64) as f32,
                y: ((1. - frac as f64) * vertp[ind1 as usize].y as f64
                    + (frac * vertp[ind2 as usize].y) as f64) as f32,
                z: zadd as f32,
            };
            imod_point_append(cont, ptadd);
        }
    }
}

/// Original: `imeshSortSurfaces` (`imesh.c:592`).
pub fn imesh_sort_surfaces(
    mesh: Option<&Imesh>,
    new_size: &mut i32,
    error: &mut i32,
) -> Option<Vec<Imesh>> {
    let mut ind: i32;
    let mut max_ind: i32;
    let mut list_inc: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;
    let mut vlow0: i32;
    let mut vlow1: i32;
    let mut vlow2: i32;
    let mut vmin: i32;
    let mut num_surf: i32;
    let mut new_ind: i32;
    let mut surf: usize;
    let mut num_ind: i32;
    let mut vmap: i32;
    let mut old_ind: i32;

    *error = 1;
    *new_size = 0;
    let mesh = mesh?;
    if mesh.list.is_empty() || mesh.vert.is_empty() {
        return None;
    }

    /* This code is adapted from 3dmod/surfpieces.cpp */

    /* Find out existing maximum index for map and check for unsupported codes */
    let list = &mesh.list;
    max_ind = -1;
    *error = 3;
    for ind in 0..mesh.list.len() {
        /* ACCUM_MAX (`b3dutil.h:40`) */
        max_ind = if max_ind > list[ind] {
            max_ind
        } else {
            list[ind]
        };
        if list[ind] == IMOD_MESH_BGNBIGPOLY || list[ind] == IMOD_MESH_BGNPOLY {
            return None;
        }
        if list[ind] > 0 && list[ind] % 2 != 0 {
            return None;
        }
    }

    /* Allocate arrays after dividing max index by 2 */
    max_ind /= 2;
    let mut vertex_map: Vec<i32> = vec![0; (max_ind + 3).max(0) as usize];
    let mut new_ind_map: Vec<i32> = vec![0; (max_ind + 3).max(0) as usize];
    let mut tlist: Vec<i32> = vec![0; mesh.list.len()];
    *error = 2;

    /* Initialize map to top index + 1 and new indexes to -1*/
    for ind in 0..=max_ind {
        vertex_map[ind as usize] = max_ind + 1;
        new_ind_map[ind as usize] = -1;
    }

    /* Copy into temporary index list, dividing by 2 */
    num_ind = 0;
    ind = 0;
    while ind < mesh.list.len() as i32 {
        if mesh.list[ind as usize] == IMOD_MESH_BGNPOLYNORM2
            || mesh.list[ind as usize] == IMOD_MESH_BGNPOLYNORM
        {
            let start = mesh.list[ind as usize];
            ind += 1;
            imod_mesh_poly_norm_factors(start, &mut list_inc, &mut vert_base, &mut norm_add);
            while mesh.list[ind as usize] != IMOD_MESH_ENDPOLY {
                tlist[num_ind as usize] = mesh.list[(ind + vert_base) as usize] / 2;
                num_ind += 1;
                ind += list_inc;
                tlist[num_ind as usize] = mesh.list[(ind + vert_base) as usize] / 2;
                num_ind += 1;
                ind += list_inc;
                tlist[num_ind as usize] = mesh.list[(ind + vert_base) as usize] / 2;
                num_ind += 1;
                ind += list_inc;
            }
        }
        ind += 1;
    }

    ind = 0;
    while ind < num_ind {
        /* For each triangle, find minimum index each vertex connects to then tie it
        together by mapping them all to lowest */
        vlow0 = min_connected_vertex(tlist[ind as usize], &mut vertex_map);
        vlow1 = min_connected_vertex(tlist[(ind + 1) as usize], &mut vertex_map);
        vlow2 = min_connected_vertex(tlist[(ind + 2) as usize], &mut vertex_map);
        vmin = if vlow0 < vlow1 { vlow0 } else { vlow1 };
        vmin = if vmin < vlow2 { vmin } else { vlow2 };
        vertex_map[vlow2 as usize] = vmin;
        vertex_map[vlow1 as usize] = vmin;
        vertex_map[vlow0 as usize] = vmin;
        ind += 3;
    }

    /* Replace with surface numbers: when a vertex maps to itself, it starts a surface,
    otherwise replace it with the value it maps to, which should already be a surface
    number */
    num_surf = 0;
    for ind in 0..=max_ind {
        vmap = vertex_map[ind as usize];
        if vmap < ind {
            vertex_map[ind as usize] = vertex_map[vmap as usize];
        } else if vmap <= max_ind {
            vertex_map[ind as usize] = num_surf;
            num_surf += 1;
        }
    }

    /* Allocate the array of meshes and then count the vertices and indexes for each */
    let mut new_mesh = match imod_meshes_new(num_surf) {
        Some(new_mesh) => new_mesh,
        None => return None,
    };
    /* The C counts into the vsize/lsize members before allocating; this
    translation keeps those counts in local vectors because the Rust `Imesh`
    derives its sizes from the `Vec` lengths. */
    let mut vcount: Vec<i32> = vec![0; num_surf.max(0) as usize];
    let mut lcount: Vec<i32> = vec![0; num_surf.max(0) as usize];
    for ind in 0..=max_ind {
        if vertex_map[ind as usize] < num_surf {
            vcount[vertex_map[ind as usize] as usize] += 2;
        }
    }
    for ind in 0..num_ind {
        old_ind = tlist[ind as usize];
        if old_ind <= max_ind && vertex_map[old_ind as usize] < num_surf {
            lcount[vertex_map[old_ind as usize] as usize] += 1;
        }
    }

    /* Allocate the arrays needed; last chance for errors.
    Rezero the list and vert sizes but add start code */
    for surf in 0..num_surf as usize {
        new_mesh[surf].surf = surf as i16;
        new_mesh[surf].list = Vec::with_capacity((lcount[surf] + 3).max(0) as usize);
        new_mesh[surf].vert = vec![Ipoint::default(); vcount[surf].max(0) as usize];
        new_mesh[surf].list.push(IMOD_MESH_BGNPOLYNORM2);
        vcount[surf] = 0;
    }

    /* Move indexes and points into meshes */
    for ind in 0..num_ind {
        old_ind = tlist[ind as usize];
        if old_ind <= max_ind && vertex_map[old_ind as usize] < num_surf {
            surf = vertex_map[old_ind as usize] as usize;
            new_ind = new_ind_map[old_ind as usize];

            /* If vertex hasn't been seen before assign it and copy vertex and normal */
            if new_ind < 0 {
                new_ind = vcount[surf];
                new_ind_map[old_ind as usize] = new_ind;
                vcount[surf] += 2;
                new_mesh[surf].vert[new_ind as usize] = mesh.vert[(2 * old_ind) as usize];
                new_mesh[surf].vert[new_ind as usize + 1] = mesh.vert[(2 * old_ind) as usize + 1];
            }

            /* Then assign new index */
            new_mesh[surf].list.push(new_ind);
        }
    }
    for surf in 0..num_surf as usize {
        /* The C leaves the allocated vertex array at its counted size and
        reports the number actually assigned in `vsize`; the Rust `Imesh`
        derives its size from the `Vec`, so trim to that count. */
        new_mesh[surf].vert.truncate(vcount[surf].max(0) as usize);
        new_mesh[surf].list.push(IMOD_MESH_ENDPOLY);
        new_mesh[surf].list.push(IMOD_MESH_END);
    }
    *error = 0;
    *new_size = num_surf;
    Some(new_mesh)
}

/// Original: `minConnectedVertex` (`imesh.c:750`), a file-static helper.
fn min_connected_vertex(vert_high: i32, vmap: &mut [i32]) -> i32 {
    let mut v1: i32;

    /* If this vertex maps to another already, walk down through mapped chain */
    let mut vert_low = vert_high;
    while vmap[vert_low as usize] < vert_low {
        vert_low = vmap[vert_low as usize];
    }

    /* Collapse chain to map all to the lowest one for efficiency of future traversals. */
    v1 = vert_high;
    while v1 > vert_low {
        let next = vmap[v1 as usize];
        vmap[v1 as usize] = vert_low;
        v1 = next;
    }
    vert_low
}

/****************************************************************************/
/* Get mesh info.                                                           */
/****************************************************************************/

/// Original: `imeshVolume` (`imesh.c:776`).
///
/// The accumulator `tvol` and the centroid sums are C `double`, but every
/// vertex expression that mixes only `float` operands stays in `float` before
/// it is widened, and `cd?` mixes a `float` vertex with the `double` centroid.
/// Those widths are reproduced exactly here.
pub fn imesh_volume(mesh: Option<&Imesh>, scale: Option<&Ipoint>, center: Option<&Ipoint>) -> f32 {
    let mut i: i32;
    let mut nsum: i32;
    let mut tvol: f64 = 0.0;
    let mut list_inc: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;
    let mut xsum: f64;
    let mut ysum: f64;
    let mut zsum: f64;
    let mut abx: f64;
    let mut aby: f64;
    let mut abz: f64;
    let mut bcx: f64;
    let mut bcy: f64;
    let mut bcz: f64;
    let mut cdx: f64;
    let mut cdy: f64;
    let mut cdz: f64;
    let mut dx: f64;
    let mut dy: f64;
    let mut dz: f64;

    let mut p1: Ipoint;
    let mut p2: Ipoint;
    let mut p3: Ipoint;
    let mut zs: f32 = 1.0;

    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return 0.0,
    };
    if mesh.list.is_empty() {
        return 0.0;
    }
    if let Some(scale) = scale {
        zs = scale.z;
    }

    /* First get centroid of mesh */
    if let Some(center) = center {
        dx = center.x as f64;
        dy = center.y as f64;
        dz = center.z as f64;
    } else {
        nsum = 0;
        xsum = 0.;
        ysum = 0.;
        zsum = 0.;
        i = 0;
        while i < mesh.list.len() as i32 {
            let code = mesh.list[i as usize];
            if code == IMOD_MESH_BGNBIGPOLY || code == IMOD_MESH_BGNPOLY {
                while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                    i += 1;
                }
            } else if code == IMOD_MESH_BGNPOLYNORM2 || code == IMOD_MESH_BGNPOLYNORM {
                let start = mesh.list[i as usize];
                i += 1;
                imod_mesh_poly_norm_factors(start, &mut list_inc, &mut vert_base, &mut norm_add);
                while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                    let p1 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                    i += list_inc;
                    let p2 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                    i += list_inc;
                    let p3 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                    i += list_inc;
                    xsum += (p1.x + p2.x + p3.x) as f64;
                    ysum += (p1.y + p2.y + p3.y) as f64;
                    zsum += (p1.z + p2.z + p3.z) as f64;
                    nsum += 3;
                }
            }
            i += 1;
        }
        if nsum == 0 {
            return 0.0;
        }
        dx = xsum / nsum as f64;
        dy = ysum / nsum as f64;
        dz = zsum / nsum as f64;
    }

    /* Add tetrahedon volumes to the centroid from each triangle */
    i = 0;
    while i < mesh.list.len() as i32 {
        let code = mesh.list[i as usize];
        if code == IMOD_MESH_BGNBIGPOLY || code == IMOD_MESH_BGNPOLY {
            while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                i += 1;
            }
        } else if code == IMOD_MESH_BGNPOLYNORM2 || code == IMOD_MESH_BGNPOLYNORM {
            let start = mesh.list[i as usize];
            i += 1;
            imod_mesh_poly_norm_factors(start, &mut list_inc, &mut vert_base, &mut norm_add);
            while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                p1 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                i += list_inc;
                p2 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                i += list_inc;
                p3 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                i += list_inc;
                abx = (p1.x - p2.x) as f64;
                aby = (p1.y - p2.y) as f64;
                abz = (zs * (p1.z - p2.z)) as f64;
                bcx = (p2.x - p3.x) as f64;
                bcy = (p2.y - p3.y) as f64;
                bcz = (zs * (p2.z - p3.z)) as f64;
                cdx = p3.x as f64 - dx;
                cdy = p3.y as f64 - dy;
                cdz = zs as f64 * (p3.z as f64 - dz);
                /* determ3 (`b3dutil.h:83`) */
                cdz = abx * bcy * cdz - abx * bcz * cdy + aby * bcz * cdx - aby * bcx * cdz
                    + abz * bcx * cdy
                    - abz * bcy * cdx;
                tvol += cdz;
            }
        }
        i += 1;
    }

    /* The sum can be negative if the centroid is not inside - so take abs */
    (tvol / 6.).abs() as f32
}

/// Original: `imeshSurfaceArea` (`imesh.c:871`).
pub fn imesh_surface_area(mesh: Option<&Imesh>, scale: Option<&Ipoint>) -> f32 {
    let mut i: i32;
    let mut tsa: f64 = 0.0;
    let mut list_inc: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;

    let mut p1: Ipoint;
    let mut p2: Ipoint;
    let mut p3: Ipoint;
    let mut p: Ipoint;
    let mut n = Ipoint::default();
    let mut n1 = Ipoint::default();
    let mut n2 = Ipoint::default();
    let mut zs: f32 = 1.0;

    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return 0.0,
    };
    if mesh.list.is_empty() {
        return 0.0;
    }
    if let Some(scale) = scale {
        zs = scale.z;
    }

    i = 0;
    while i < mesh.list.len() as i32 {
        let code = mesh.list[i as usize];
        if code == IMOD_MESH_BGNBIGPOLY || code == IMOD_MESH_BGNPOLY {
            n.x = 0.0;
            n.y = 0.0;
            n.z = 0.0;
            i += 1;
            if mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                p = mesh.vert[mesh.list[i as usize] as usize];
                p1 = p;
                i += 1;
                while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                    p2 = mesh.vert[mesh.list[i as usize] as usize];
                    n.x += (p1.y * p2.z * zs) - (p1.z * zs * p2.y);
                    n.y += (p1.z * zs * p2.x) - (p1.x * p2.z * zs);
                    n.z += (p1.x * p2.y) - (p1.y * p2.x);
                    p1 = p2;
                    i += 1;
                }
                p2 = p;
                n.x += (p1.y * p2.z * zs) - (p1.z * zs * p2.y);
                n.y += (p1.z * zs * p2.x) - (p1.x * p2.z * zs);
                n.z += (p1.x * p2.y) - (p1.y * p2.x);
                tsa += ((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5f32 as f64;
            }
        } else if code == IMOD_MESH_BGNPOLYNORM2 || code == IMOD_MESH_BGNPOLYNORM {
            let start = mesh.list[i as usize];
            i += 1;
            imod_mesh_poly_norm_factors(start, &mut list_inc, &mut vert_base, &mut norm_add);
            while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                n.x = 0.0;
                n.y = 0.0;
                n.z = 0.0;
                p1 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                i += list_inc;
                p2 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                i += list_inc;
                p3 = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                i += list_inc;

                n1.x = p1.x - p2.x;
                n1.y = p1.y - p2.y;
                n1.z = (p1.z - p2.z) * zs;
                n2.x = p3.x - p2.x;
                n2.y = p3.y - p2.y;
                n2.z = (p3.z - p2.z) * zs;
                imod_point_cross(&n1, &n2, &mut n);

                tsa += ((n.x * n.x + n.y * n.y + n.z * n.z) as f64).sqrt() * 0.5;
            }
        } else if code == IMOD_MESH_END {
            return tsa as f32;
        }
        i += 1;
    }
    tsa as f32
}

/// Original: `imodMeshGetBBox` (`imesh.c:953`).
///
/// Source defect, translated as written: the `IMOD_MESH_BGNBIGPOLY` /
/// `IMOD_MESH_BGNPOLY` branch never advances `i` inside its `while`, so a mesh
/// carrying either code makes the C spin forever.  No in-scope caller reaches
/// it, and no substitute is invented here.
pub fn imod_mesh_get_bbox(mesh: Option<&Imesh>, ll: &mut Ipoint, ur: &mut Ipoint) -> i32 {
    let mut pt: Ipoint;
    let mut list_inc: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;
    let mut i: i32;

    let mesh = match mesh {
        Some(mesh) => mesh,
        None => return -1,
    };
    if mesh.list.is_empty() || mesh.vert.is_empty() {
        return -1;
    }
    ll.x = 1.0e30;
    ll.y = 1.0e30;
    ll.z = 1.0e30;
    ur.x = -1.0e30;
    ur.y = -1.0e30;
    ur.z = -1.0e30;

    i = 0;
    while i < mesh.list.len() as i32 {
        let code = mesh.list[i as usize];
        if code == IMOD_MESH_BGNBIGPOLY || code == IMOD_MESH_BGNPOLY {
            i += 1;
            while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                pt = mesh.vert[mesh.list[i as usize] as usize];
                ll.x = if ll.x < pt.x { ll.x } else { pt.x };
                ll.y = if ll.y < pt.y { ll.y } else { pt.y };
                ll.z = if ll.z < pt.z { ll.z } else { pt.z };
                ur.x = if ur.x > pt.x { ur.x } else { pt.x };
                ur.y = if ur.y > pt.y { ur.y } else { pt.y };
                ur.z = if ur.z > pt.z { ur.z } else { pt.z };
            }
        } else if code == IMOD_MESH_BGNPOLYNORM2 || code == IMOD_MESH_BGNPOLYNORM {
            let start = mesh.list[i as usize];
            i += 1;
            imod_mesh_poly_norm_factors(start, &mut list_inc, &mut vert_base, &mut norm_add);
            while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                for _j in 0..3 {
                    pt = mesh.vert[mesh.list[(i + vert_base) as usize] as usize];
                    i += list_inc;
                    ll.x = if ll.x < pt.x { ll.x } else { pt.x };
                    ll.y = if ll.y < pt.y { ll.y } else { pt.y };
                    ll.z = if ll.z < pt.z { ll.z } else { pt.z };
                    ur.x = if ur.x > pt.x { ur.x } else { pt.x };
                    ur.y = if ur.y > pt.y { ur.y } else { pt.y };
                    ur.z = if ur.z > pt.z { ur.z } else { pt.z };
                }
            }
        }
        i += 1;
    }
    if ll.x > 1.0e29 {
        return -1;
    }
    0
}

/*******************
 * Meshing parameter functions
 ******************/

/// Original: `imeshParamsNew` (`imesh.c:1011`).
pub fn imesh_params_new() -> Option<MeshParams> {
    let mut params = MeshParams::default();
    imesh_params_default(&mut params);
    Some(params)
}

/// Original: `imeshParamsDefault` (`imesh.c:1023`).
pub fn imesh_params_default(params: &mut MeshParams) {
    params.flags = IMESH_MK_NORM | IMESH_MK_TIME;
    params.cap = IMESH_CAP_OFF;
    params.passes = 1;
    params.cap_skip_nz = 0;
    params.incz_low_res = 4;
    params.incz_high_res = 1;
    params.minz = DEFAULT_VALUE;
    params.maxz = DEFAULT_VALUE;
    params.xmin = -DEFAULT_FLOAT;
    params.xmax = DEFAULT_FLOAT;
    params.ymin = -DEFAULT_FLOAT;
    params.ymax = DEFAULT_FLOAT;
    params.overlap = 0.;
    params.tube_diameter = 10.;
    params.tol_low_res = 2.0;
    params.tol_high_res = 0.25;
    params.flat_crit = 1.5;
    params.spare_int = 0;
    params.spare_float = 0.;
    params.cap_skip_zlist = None;
}

/// Original: `imeshParamsDup` (`imesh.c:1051`).
pub fn imesh_params_dup(params: Option<&MeshParams>) -> Option<MeshParams> {
    let params = params?;
    let mut newpar = imesh_params_new()?;
    newpar = params.clone();
    newpar.cap_skip_zlist = None;
    newpar.cap_skip_nz = 0;
    let mut lto = newpar.cap_skip_zlist.take();
    let mut nto = newpar.cap_skip_nz;
    if imesh_copy_skip_list(
        params.cap_skip_zlist.as_deref(),
        params.cap_skip_nz,
        &mut lto,
        &mut nto,
    ) != 0
    {
        return None;
    }
    newpar.cap_skip_zlist = lto;
    newpar.cap_skip_nz = nto;
    Some(newpar)
}

/// Original: `imeshParamsDelete` (`imesh.c:1074`).
pub fn imesh_params_delete(params: Option<MeshParams>) {
    drop(params);
}

/// Original: `imeshCopySkipList` (`imesh.c:1090`).
pub fn imesh_copy_skip_list(
    lfrom: Option<&[i32]>,
    nfrom: i32,
    lto: &mut Option<Vec<i32>>,
    nto: &mut i32,
) -> i32 {
    if lto.is_some() && *nto != 0 {
        *lto = None;
    }
    *nto = 0;
    if let Some(lfrom) = lfrom {
        /* The `b3dError(stderr, "Error getting memory to copy cap exclusion
        list")` branch of the source fires only when this allocation fails,
        which this translation cannot reach. */
        let mut list: Vec<i32> = vec![0; (nfrom + 3).max(0) as usize];
        for i in 0..nfrom as usize {
            list[i] = lfrom[i];
        }
        *lto = Some(list);
        *nto = nfrom;
    }
    0
}
