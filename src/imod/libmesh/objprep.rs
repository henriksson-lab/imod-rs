//! Translation of `IMOD/libmesh/objprep.c` -- analysis, preparation and
//! skinning of an object, plus the contour duplication and preparation
//! routines declared by `IMOD/include/mkmesh.h`.

use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_error};
use crate::imod::libimod::icont::{
    imod_contour_clear, imod_contour_copy, imod_contour_delete, imod_contour_dup,
    imod_contour_fit_plane, imod_contour_get_bbox, imod_contour_length, imod_contour_reduce,
    imod_contour_z_value,
};
use crate::imod::libimod::imat::{
    B3D_X, B3D_Y, Imat, imod_mat_id, imod_mat_inverse, imod_mat_new, imod_mat_rot, imod_mat_scale,
    imod_mat_trans, imod_mat_transform,
};
use crate::imod::libimod::imesh::{
    DEFAULT_FLOAT, DEFAULT_VALUE, IMESH_MK_IS_COPY, IMESH_MK_TUBE, IMESH_MK_USE_MEAN, imesh_resol,
    imod_meshes_delete, imodel_mesh_add,
};
use crate::imod::libimod::imodel::{ICONT_OPEN, Icont, Imesh, Iobj, Ipoint};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_PNT_NOMODV, imod_object_add_contour, imod_object_copy, imod_object_new,
    imod_object_remove_contour, iobj_close, iobj_open,
};
use crate::imod::libimod::ipoint::{
    imod_point_add, imod_point_append, imod_point_distance, imod_point_dot, imod_point_normalize,
};
use crate::imod::libimod::istore::{
    GEN_STORE_GAP, GEN_STORE_ONEPOINT, Istore, istore_copy_cont_surf_items, istore_copy_non_index,
    istore_insert, istore_point_is_gap,
};
use crate::imod::libmesh::mkmesh::imesh_set_min_max;
use crate::imod::libmesh::skinobj::imesh_skin_object;

use crate::imod::libimod::icont::ICONT_TEMPUSE;

/// Original: `analyzePrepSkinObj` (`objprep.c:38`).
///
/// Provides a single call for analysis, preparation, and skinning of object
/// `obj`.  Uses the meshing parameters in the `meshParam` member of `obj`.
/// Duplicates the contours unless `IMESH_MK_IS_COPY` is set in `flags`.
/// Analyzes for flatness if `flatCrit` is nonzero and if the contours are
/// not flat enough, finds a separate rotation to flatness for each surface.
/// Calls `imeshPrepContours` and `imeshSkinObject`.  Returns 1 for error.
pub fn analyze_prep_skin_obj(
    obj: &mut Iobj,
    resol: i32,
    scale: &Ipoint,
    in_cb: Option<fn(i32) -> i32>,
) -> i32 {
    let mut tri_min = Ipoint {
        x: -DEFAULT_FLOAT,
        y: -DEFAULT_FLOAT,
        z: -DEFAULT_FLOAT,
    };
    let mut tri_max = Ipoint {
        x: DEFAULT_FLOAT,
        y: DEFAULT_FLOAT,
        z: DEFAULT_FLOAT,
    };
    let mut subsets = 0;
    let mut maxsurf = 0;
    let mut max_zdiff = 0.0f32;
    let mut bbmin: Option<Vec<Ipoint>> = None;
    let mut bbmax: Option<Vec<Ipoint>> = None;
    let mut volume: Option<Vec<f32>> = None;
    let mut volsort: Option<Vec<f32>> = None;
    let mut mat: Option<Imat> = None;
    let mut inv: Option<Imat> = None;
    let mut cen = Ipoint::default();
    let mut cnorm = Ipoint::default();
    let mut normsum = Ipoint::default();
    let mut ceninv = Ipoint::default();
    let mut sclinv = Ipoint::default();
    let mut ref_norm = Ipoint::default();
    let mut sclskin: Ipoint;
    let mut beta: f64;
    let mut alpha: f64;
    let mut zrot: f64 = 0.;
    let dtor: f64 = 0.017453293;
    let mut dval = 0.0f32;
    let small_val = 1.0e-4f32;

    /* Unpack values from the mesh parameter structure */
    let Some(params) = obj.mesh_param.as_ref() else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("ERROR: analyzePrepSkinObj - no meshing parameters in object"),
        );
        return 1;
    };
    let flags = params.flags;
    let cap = params.cap;
    let skip_passes = params.passes;
    let incz;
    let tol;
    if resol != 0 {
        incz = params.incz_low_res;
        tol = params.tol_low_res;
    } else {
        incz = params.incz_high_res;
        tol = params.tol_high_res;
    }
    let minz = params.minz;
    let maxz = params.maxz;
    tri_min.x = params.xmin;
    tri_max.x = params.xmax;
    tri_min.y = params.ymin;
    tri_max.y = params.ymax;
    let overlap = params.overlap;
    let tube_diameter = params.tube_diameter;
    let flat_crit = params.flat_crit;
    let cap_skip_zlist: Option<Vec<i32>> = params.cap_skip_zlist.clone();
    let cap_skip_nz = params.cap_skip_nz;
    imesh_set_min_max(tri_min, tri_max);
    let make_tubes = i32::from(iobj_open(obj.flags) != 0 && (flags & IMESH_MK_TUBE) != 0);
    if make_tubes != 0 && tube_diameter < 0. && tube_diameter >= -1.0001 {
        obj.flags |= IMOD_OBJFLAG_PNT_NOMODV;
    } else {
        obj.flags &= !IMOD_OBJFLAG_PNT_NOMODV;
    }
    if flat_crit > 0. && make_tubes == 0 {
        /* Get arrays for bounding box and find biggest Z difference in contours */
        let mut bbmin_v = vec![Ipoint::default(); obj.cont.len()];
        let mut bbmax_v = vec![Ipoint::default(); obj.cont.len()];
        let mut volume_v = vec![0.0f32; obj.cont.len()];
        let volsort_v = vec![0.0f32; obj.cont.len()];
        let mat_v = imod_mat_new(3);
        if mat_v.is_none() {
            b3d_error(
                Some(&mut ImodFile::Stderr),
                format_args!("Memory error allocating boundary box arrays\n"),
            );
            clean_prep_arrays(None, None, None, None, None, None, None);
            return 1;
        }
        for co in 0..obj.cont.len() {
            let cont = &obj.cont[co];
            if !cont.pts.is_empty() {
                maxsurf = maxsurf.max(cont.surf);
                imod_contour_get_bbox(Some(cont), &mut bbmin_v[co], &mut bbmax_v[co]);
                max_zdiff = if max_zdiff < bbmax_v[co].z - bbmin_v[co].z {
                    bbmax_v[co].z - bbmin_v[co].z
                } else {
                    max_zdiff
                };

                /* Add a little thickness to the differences to avoid zero volumes */
                volume_v[co] = (bbmax_v[co].x + 1. - bbmin_v[co].x)
                    * (bbmax_v[co].y + 1. - bbmin_v[co].y)
                    * (bbmax_v[co].z + 1. - bbmin_v[co].z);
            }
        }

        if max_zdiff >= flat_crit {
            subsets = 1;
            bbmin = Some(bbmin_v);
            bbmax = Some(bbmax_v);
            volume = Some(volume_v);
            volsort = Some(volsort_v);
            mat = mat_v;
        } else {
            clean_prep_arrays(
                Some(bbmin_v),
                Some(bbmax_v),
                Some(volume_v),
                Some(volsort_v),
                mat_v,
                inv.take(),
                None,
            );
        }
    }

    if subsets == 0 {
        /* Simple case of no tilted contours */
        let mut dup: Option<Iobj> = if flags & IMESH_MK_IS_COPY != 0 {
            None
        } else {
            match imesh_dup_marked_conts(obj, 0) {
                Some(o) => Some(o),
                None => return 1,
            }
        };
        {
            let use_obj: &mut Iobj = match dup.as_mut() {
                Some(o) => o,
                None => obj,
            };
            if make_tubes == 0
                && imesh_prep_contours(
                    use_obj,
                    minz,
                    maxz,
                    incz,
                    tol,
                    (flags & IMESH_MK_USE_MEAN) as i32,
                ) != 0
            {
                return 1;
            }
            if imesh_skin_object(
                use_obj,
                scale,
                overlap as f64,
                cap,
                cap_skip_zlist.as_deref(),
                cap_skip_nz,
                incz,
                flags,
                skip_passes,
                tube_diameter as f64,
                in_cb,
            ) != 0
            {
                return 1;
            }
        }

        if let Some(mut use_obj) = dup.take() {
            if !obj.mesh.is_empty() {
                let size = obj.mesh.len() as i32;
                imod_meshes_delete(Some(std::mem::take(&mut obj.mesh)), size);
            }
            obj.mesh = std::mem::take(&mut use_obj.mesh);
        }
        return 0;
    }

    /* Tilted contours were found.  Now loop on surfaces */
    if !obj.mesh.is_empty() {
        let size = obj.mesh.len() as i32;
        imod_meshes_delete(Some(std::mem::take(&mut obj.mesh)), size);
    }
    obj.mesh = Vec::new();

    let bbmin_v = bbmin.as_mut().unwrap();
    let bbmax_v = bbmax.as_mut().unwrap();
    let volume_v = volume.as_mut().unwrap();
    let volsort_v = volsort.as_mut().unwrap();
    let mut mat_v = mat.take().unwrap();

    for surf in 0..=maxsurf {
        /* Compute mean volume and centroid of all the bounding boxes */
        let mut volsum = 0.0f32;
        let mut nin_surf = 0usize;
        cen.x = 0.;
        cen.y = 0.;
        cen.z = 0.;
        for co in 0..obj.cont.len() {
            let cont = &obj.cont[co];
            if cont.surf == surf && !cont.pts.is_empty() {
                volsum += volume_v[co];
                cen.x += (volume_v[co] as f64 * (bbmin_v[co].x + bbmax_v[co].x) as f64 / 2.) as f32;
                cen.y += (volume_v[co] as f64 * (bbmin_v[co].y + bbmax_v[co].y) as f64 / 2.) as f32;
                cen.z += (volume_v[co] as f64 * (bbmin_v[co].z + bbmax_v[co].z) as f64 / 2.) as f32;
                volsort_v[nin_surf] = volume_v[co];
                nin_surf += 1;
            }
        }

        if nin_surf < 2 {
            continue;
        }

        cen.x /= volsum;
        cen.y /= volsum;
        cen.z /= volsum;
        let _volavg = volsum / nin_surf as f32;

        volsort_v[..nin_surf].sort_by(floatcmp_ord);
        let volmed = volsort_v[nin_surf / 2];

        let mut num_norm = 0;
        normsum.x = 0.;
        normsum.y = 0.;
        normsum.z = 0.;
        for co in 0..obj.cont.len() {
            let cont = &obj.cont[co];
            if cont.surf == surf && cont.pts.len() > 2 && volume_v[co] >= volmed {
                let mut alpha_l = 0.0f64;
                let mut beta_l = 0.0f64;
                if imod_contour_fit_plane(
                    cont,
                    scale,
                    &mut cnorm,
                    &mut dval,
                    &mut alpha_l,
                    &mut beta_l,
                ) == 0
                {
                    /* Invert the normal if it does not match the reference */
                    if (num_norm != 0 && imod_point_dot(&cnorm, &ref_norm) < 0.)
                        || (num_norm == 0 && cnorm.z < 0.)
                    {
                        cnorm.x = -cnorm.x;
                        cnorm.y = -cnorm.y;
                        cnorm.z = -cnorm.z;
                    }
                    normsum.x += cnorm.x;
                    normsum.y += cnorm.y;
                    normsum.z += cnorm.z;
                    if num_norm == 0 {
                        ref_norm = cnorm;
                    }
                    num_norm += 1;
                }
            }
        }

        /* Get mean of the normals then find the angles to rotate plane flat */
        alpha = 0.;
        beta = 0.;
        if num_norm != 0 {
            cnorm.x = normsum.x / num_norm as f32;
            cnorm.y = normsum.y / num_norm as f32;
            cnorm.z = normsum.z / num_norm as f32;

            if (cnorm.x as f64).abs() > small_val as f64
                || (cnorm.z as f64).abs() > small_val as f64
            {
                beta = -(cnorm.x as f64).atan2(cnorm.z as f64);
            }
            zrot = cnorm.z as f64 * beta.cos() - cnorm.x as f64 * beta.sin();
            alpha = -(zrot.atan2(cnorm.y as f64) - 1.570796);
        }

        /* Figure out how much the normals between planes are being compressed by
        Z-scaling - compute the normal in the original space and transform it */
        sclinv.x = 1. / scale.x;
        sclinv.y = 1. / scale.y;
        sclinv.z = 1. / scale.z;
        sclskin = *scale;
        cnorm.x *= scale.x;
        cnorm.y *= scale.y;
        cnorm.z *= scale.z;
        imod_point_normalize(&mut cnorm);

        imod_mat_id(&mut mat_v);
        imod_mat_scale(&mut mat_v, scale);
        imod_mat_rot(&mut mat_v, beta / dtor, B3D_Y);
        imod_mat_rot(&mut mat_v, alpha / dtor, B3D_X);
        imod_mat_scale(&mut mat_v, &sclinv);
        imod_mat_transform(&mat_v, &cnorm, &mut ref_norm);

        /* The Z height of this normal is the distance between planes after
        flattening so the inverse is the amount that the z scaling needs to
        increase when flattening the contours, while the skinning Z scale needs
        to change to accommodate */
        sclinv.z /= ref_norm.z;
        sclskin.z *= ref_norm.z;

        /* Compose the transformation */
        ceninv.x = -cen.x;
        ceninv.y = -cen.y;
        ceninv.z = -cen.z;
        imod_mat_id(&mut mat_v);
        imod_mat_trans(&mut mat_v, &ceninv);
        imod_mat_scale(&mut mat_v, scale);
        imod_mat_rot(&mut mat_v, beta / dtor, B3D_Y);
        imod_mat_rot(&mut mat_v, alpha / dtor, B3D_X);
        imod_mat_scale(&mut mat_v, &sclinv);
        imod_mat_trans(&mut mat_v, &cen);

        /* Mark the contours for duplication and get object*/
        for co in 0..obj.cont.len() {
            if obj.cont[co].surf == surf {
                obj.cont[co].flags |= ICONT_TEMPUSE;
            }
        }
        let Some(mut use_obj) = imesh_dup_marked_conts(obj, ICONT_TEMPUSE) else {
            clean_prep_arrays(
                bbmin.take(),
                bbmax.take(),
                volume.take(),
                volsort.take(),
                Some(mat_v),
                inv.take(),
                None,
            );
            return 1;
        };

        /* Transform the contours */
        for co in 0..use_obj.cont.len() {
            let mut zmin = 1.0e10f32;
            let mut zmax = -1.0e10f32;
            for pt in 0..use_obj.cont[co].pts.len() {
                let src = use_obj.cont[co].pts[pt];
                imod_mat_transform(&mat_v, &src, &mut cnorm);
                use_obj.cont[co].pts[pt] = cnorm;
                zmin = if zmin > cnorm.z { cnorm.z } else { zmin };
                zmax = if zmax < cnorm.z { cnorm.z } else { zmax };
            }
            let _ = (zmin, zmax);
        }
        if imesh_prep_contours(&mut use_obj, minz, maxz, incz, tol, 1) != 0 {
            clean_prep_arrays(
                bbmin.take(),
                bbmax.take(),
                volume.take(),
                volsort.take(),
                Some(mat_v),
                inv.take(),
                Some(use_obj),
            );
            return 1;
        }

        /* Evaluate whether a Z offset is needed to keep contours from rounding
        to inappropriate Z values */
        let mut zofs = 0.0f32;
        let mut minofs = 0.0f32;
        let mut minbad = (use_obj.cont.len() * use_obj.cont.len()) as i32;
        let mut nbad;
        while zofs < 0.5 && zofs > -0.5 {
            nbad = 0;

            for co in 0..use_obj.cont.len().saturating_sub(1) {
                let z1 = use_obj.cont[co].pts[0].z as f64;
                for co2 in co + 1..use_obj.cont.len() {
                    let z2 = use_obj.cont[co2].pts[0].z as f64;
                    if ((z1 - z2).abs() + 0.5).floor() as i32 == 1 {
                        let izdiff = (z1 + zofs as f64 + 0.5).floor() as i32
                            - (z2 + zofs as f64 + 0.5).floor() as i32;
                        if izdiff != 1 && izdiff != -1 {
                            nbad += 1;
                        }
                    }
                }
            }
            if nbad < minbad {
                minbad = nbad;
                minofs = zofs;
            }
            if nbad == 0 {
                break;
            }
            if zofs <= 0. {
                zofs = -zofs + 0.05;
            } else {
                zofs = -zofs;
            }
        }

        /* If we found an offset that minimizes or eliminates the problem,
        apply it.  Wait and see if people need anything fancier */
        if minofs != 0. {
            for co in 0..use_obj.cont.len() {
                for pt in 0..use_obj.cont[co].pts.len() {
                    use_obj.cont[co].pts[pt].z += minofs;
                }
            }

            /* Add shift to transformation so it will be taken out in inverse */
            cnorm.x = 0.;
            cnorm.y = 0.;
            cnorm.z = minofs;
            imod_mat_trans(&mut mat_v, &cnorm);
        }

        /* Skin and add to object mesh */
        if imesh_skin_object(
            &mut use_obj,
            &sclskin,
            overlap as f64,
            cap,
            cap_skip_zlist.as_deref(),
            cap_skip_nz,
            incz,
            flags,
            skip_passes,
            tube_diameter as f64,
            None,
        ) != 0
        {
            clean_prep_arrays(
                bbmin.take(),
                bbmax.take(),
                volume.take(),
                volsort.take(),
                Some(mat_v),
                inv.take(),
                Some(use_obj),
            );
            return 1;
        }

        /* Need a normal transform as well as the point transform */
        inv = imod_mat_inverse(&mat_v);
        imod_mat_id(&mut mat_v);
        imod_mat_rot(&mut mat_v, -alpha / dtor, B3D_X);
        imod_mat_rot(&mut mat_v, -beta / dtor, B3D_Y);

        /* Transform the mesh points and the normals, then add to object */
        for m in 0..use_obj.mesh.len() {
            let mut i = 0usize;
            while i < use_obj.mesh[m].vert.len() {
                let src = use_obj.mesh[m].vert[i];
                imod_mat_transform(inv.as_ref().unwrap(), &src, &mut cnorm);
                use_obj.mesh[m].vert[i] = cnorm;
                let src = use_obj.mesh[m].vert[i + 1];
                imod_mat_transform(&mat_v, &src, &mut cnorm);
                imod_point_normalize(&mut cnorm);
                use_obj.mesh[m].vert[i + 1] = cnorm;
                i += 2;
            }

            let mesh: Imesh = use_obj.mesh[m].clone();
            imodel_mesh_add(Some(&mesh), &mut obj.mesh);
        }
        inv = None;

        use_obj.mesh = Vec::new();
    }
    clean_prep_arrays(
        bbmin.take(),
        bbmax.take(),
        volume.take(),
        volsort.take(),
        Some(mat_v),
        inv.take(),
        None,
    );
    0
}

/// Original static `cleanPrepArrays` (`objprep.c:404`).
///
/// Cleans up all possible arrays set up for flattening analysis.  The C
/// `free`/`imodMatDelete`/`imodObjectDelete` calls become ownership transfers
/// into this function, which drops them.
pub fn clean_prep_arrays(
    bbmin: Option<Vec<Ipoint>>,
    bbmax: Option<Vec<Ipoint>>,
    volume: Option<Vec<f32>>,
    volsort: Option<Vec<f32>>,
    mat: Option<Imat>,
    inv: Option<Imat>,
    use_obj: Option<Iobj>,
) {
    drop(bbmin);
    drop(bbmax);
    drop(volume);
    drop(volsort);
    drop(mat);
    drop(inv);
    drop(use_obj);
}

/// Original static `floatcmp` (`objprep.c:423`).
pub fn floatcmp(f1: f32, f2: f32) -> i32 {
    if f1 < f2 {
        return -1;
    }
    if f1 > f2 {
        return 1;
    }
    0
}

/// `floatcmp` as an ordering, for the source's `qsort` call.
fn floatcmp_ord(f1: &f32, f2: &f32) -> std::cmp::Ordering {
    floatcmp(*f1, *f2).cmp(&0)
}

/// Original: `imodMeshesDeleteRes` (`objprep.c:438`).
///
/// Deletes meshes in `meshp` and their `store` elements if their resolution
/// flag matches `resol`.  Returns -1 for error.
pub fn imod_meshes_delete_res(meshp: &mut Vec<Imesh>, size: &mut i32, resol: i32) -> i32 {
    let mut newsize = 0usize;
    if meshp.is_empty() {
        return -1;
    }

    let mut ms = 0usize;
    while ms < *size as usize {
        if imesh_resol(meshp[ms].flag) == resol {
            meshp[ms].vert = Vec::new();
            meshp[ms].list = Vec::new();
            meshp[ms].store = Vec::new();
        } else {
            meshp.swap(newsize, ms);
            newsize += 1;
        }
        ms += 1;
    }
    meshp.truncate(newsize);
    *size = newsize as i32;
    0
}

/// Original: `imeshPrepContours` (`objprep.c:472`).
///
/// Flattens, resections, reduces resolution, and cleans up small numbers in
/// the contours of `obj`.  Returns 1 for error.
pub fn imesh_prep_contours(
    obj: &mut Iobj,
    minz: i32,
    maxz: i32,
    incz: i32,
    tol: f32,
    use_mean_z: i32,
) -> i32 {
    for co in 0..obj.cont.len() {
        let cont = &mut obj.cont[co];
        let mut zval = cont.pts[0].z;
        if use_mean_z != 0 {
            for i in 1..cont.pts.len() {
                zval += cont.pts[i].z;
            }
            zval /= cont.pts.len() as f32;
        }
        for i in 1..cont.pts.len() {
            cont.pts[i].z = zval;
        }
    }

    /* Add phantom ends to open contours if possible */
    if iobj_close(obj.flags) != 0 && extend_open_ends(obj) != 0 {
        return 1;
    }

    if resecobj(obj, minz, maxz, incz) != 0 {
        return 1;
    }
    if reduce_obj(obj, tol) != 0 {
        return 1;
    }
    cleanzero(obj);
    0
}

/// Original static `resecobj` (`objprep.c:501`).
///
/// Removes contours that don't fall on a multiple of `incz`.
pub fn resecobj(obj: &mut Iobj, minz: i32, maxz: i32, mut incz: i32) -> i32 {
    let mut z = 0;

    if incz <= 0 {
        incz = 1;
    }
    if minz == DEFAULT_VALUE && maxz == DEFAULT_VALUE && incz == 1 {
        return 0;
    }

    let mut co: i32 = 0;
    while co < obj.cont.len() as i32 {
        let mut rmcont = false;

        if obj.cont[co as usize].pts.is_empty() {
            rmcont = true;
        }

        /*
         * find the zvalue of the contour.
         */
        if !rmcont {
            z = imod_contour_z_value(Some(&obj.cont[co as usize]));
        }

        /*
         * Check the Z value.
         */
        if (minz != DEFAULT_VALUE && z < minz)
            || (maxz != DEFAULT_VALUE && z > maxz)
            || (z % incz) != 0
        {
            rmcont = true;
        }

        if rmcont {
            imod_contour_clear(&mut obj.cont[co as usize]);
            if imod_object_remove_contour(obj, co) != 0 {
                b3d_error(
                    Some(&mut ImodFile::Stderr),
                    format_args!("Error removing contour at unneeded slice in Z\n"),
                );
                return 1;
            }
            co -= 1;
        }
        co += 1;
    }
    0
}

/// Original static `ReduceObj` (`objprep.c:548`).
///
/// Reduces points by removing ones within `dist` of the remaining lines.
pub fn reduce_obj(obj: &mut Iobj, dist: f32) -> i32 {
    if dist <= 0.0 {
        return 0;
    }

    for co in 0..obj.cont.len() {
        let mut tol = dist;

        /* Check for a loopback contour, end points matching start points */
        let psize = obj.cont[co].pts.len() as i32;
        let num_test = 4.min(psize / 2 - 1);
        let mut loop_back = 1;
        for pt in 1..=num_test {
            let cont = &obj.cont[co];
            if ((cont.pts[pt as usize].x - cont.pts[(psize - pt) as usize].x) as f64).abs() > 0.001
                || ((cont.pts[pt as usize].y - cont.pts[(psize - pt) as usize].y) as f64).abs()
                    > 0.001
            {
                loop_back = 0;
                break;
            }
        }

        if psize > 4 && loop_back == 0 {
            while tol > 0.01 * dist {
                let Some(mut tc) = imod_contour_dup(&obj.cont[co]) else {
                    b3d_error(
                        Some(&mut ImodFile::Stderr),
                        format_args!("Failed to get duplicate contour for point reduction\n"),
                    );
                    return 1;
                };
                imod_contour_reduce(Some(&mut tc), tol);
                if tc.pts.len() < 4 {
                    tol *= 0.5;
                    imod_contour_delete(&mut tc);
                } else {
                    imod_contour_clear(&mut obj.cont[co]);
                    imod_contour_copy(&tc, &mut obj.cont[co]);
                    break;
                }
            }
        }
    }
    0
}

/// Original static `extendOpenEnds` (`objprep.c:596`).
pub fn extend_open_ends(obj: &mut Iobj) -> i32 {
    let mut dzmin: i32 = 0;
    let mut retval = 0;
    let mut num_phant = 0usize;
    let gap_length_ratio = 0.33f32;
    let mut store = Istore::default();

    let mut phant_conts: Vec<i32> = vec![0; obj.cont.len()];
    let mut contz: Vec<i32> = vec![0; obj.cont.len()];

    /* Make a list of contours with phantom ends: >= 2 gaps on either end */
    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        contz[co] = imod_contour_z_value(Some(cont));
        if (cont.flags & ICONT_OPEN) != 0 && cont.pts.len() > 1 {
            let start_gap = istore_point_is_gap(&cont.store, 0);
            let end_gap = istore_point_is_gap(&cont.store, cont.pts.len() as i32 - 2);
            if (start_gap != 0 && end_gap != 0)
                || (start_gap != 0 && istore_point_is_gap(&cont.store, 1) != 0)
                || (end_gap != 0
                    && istore_point_is_gap(&cont.store, cont.pts.len() as i32 - 3) != 0)
            {
                phant_conts[num_phant] = co as i32;
                num_phant += 1;
            }
        }
    }
    if num_phant == 0 {
        return 0;
    }

    for co in 0..obj.cont.len() {
        /* Extend a contour if it is open and neither end has phantom points and
        the distance from end to start is greater than a fraction of the total
        contour length */
        let extend = {
            let cont = &obj.cont[co];
            (cont.flags & ICONT_OPEN) != 0
                && cont.pts.len() > 1
                && istore_point_is_gap(&cont.store, 0) == 0
                && istore_point_is_gap(&cont.store, cont.pts.len() as i32 - 2) == 0
                && imod_point_distance(&cont.pts[0], &cont.pts[cont.pts.len() - 1])
                    > gap_length_ratio * imod_contour_length(Some(cont), 0)
        };
        if extend {
            let mut nco: i32 = -1;
            for phan in 0..num_phant {
                let mut dz = contz[phant_conts[phan] as usize] - contz[co];
                if dz < 0 {
                    dz = -dz;
                }
                if nco < 0 || dz < dzmin {
                    dzmin = dz;
                    nco = phant_conts[phan];
                }
            }

            let nearco = obj.cont[nco as usize].clone();

            /* Find last gap point at start and first point after gap at end */
            let mut pt_start: i32 = -1;
            while istore_point_is_gap(&nearco.store, pt_start + 1) != 0
                && pt_start < nearco.pts.len() as i32 / 2
            {
                pt_start += 1;
            }
            let mut pt_end: i32 = nearco.pts.len() as i32;
            while istore_point_is_gap(&nearco.store, pt_end - 2) != 0
                && pt_end > nearco.pts.len() as i32 / 2
            {
                pt_end -= 1;
            }

            /* Add points at start and mark as gaps */
            store.flags = GEN_STORE_ONEPOINT;
            store.type_ = GEN_STORE_GAP;
            store.value.set_i(0);
            for pt in 0..=pt_start {
                let mut pnt = nearco.pts[pt as usize];
                pnt.z = contz[co] as f32;
                if imod_point_add(&mut obj.cont[co], Some(pnt), pt) == 0 {
                    retval = 1;
                    break;
                }
                store.index.set_i(pt);
                if istore_insert(&mut obj.cont[co].store, store) != 0 {
                    retval = 1;
                    break;
                }
            }

            /* Add points at end and mark each preceding one as a gap */
            for pt in pt_end..nearco.pts.len() as i32 {
                let mut pnt = nearco.pts[pt as usize];
                pnt.z = contz[co] as f32;
                if imod_point_append(&mut obj.cont[co], pnt) == 0 {
                    retval = 1;
                    break;
                }
                store.index.set_i(obj.cont[co].pts.len() as i32 - 2);
                if istore_insert(&mut obj.cont[co].store, store) != 0 {
                    retval = 1;
                    break;
                }
            }
        }
    }
    if retval != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("An error occurred extending the open ends\n"),
        );
    }
    retval
}

/// Original static `cleanzero` (`objprep.c:712`).
///
/// Avoids underflow exceptions.  As in the source, this copies each point into
/// a local and modifies the local, so it has no effect on the object.
pub fn cleanzero(obj: &mut Iobj) {
    for co in 0..obj.cont.len() {
        for pt in 0..obj.cont[co].pts.len() {
            let mut pnt = obj.cont[co].pts[pt];

            if (pnt.x < 0.001f32) && (pnt.x > -0.001f32) {
                pnt.x = 0.0f32;
            }
            if (pnt.y < 0.001f32) && (pnt.y > -0.001f32) {
                pnt.y = 0.0f32;
            }
            if (pnt.z < 0.001f32) && (pnt.z > -0.001f32) {
                pnt.z = 0.0f32;
            }
        }
    }
}

/// Original: `imeshDupMarkedConts` (`objprep.c:740`).
///
/// Makes an object with duplicates of the contours in `obj`; either contours
/// whose flag has a non-zero AND with `flag`, or all non-empty contours if
/// `flag` is zero.  Copies store data but not meshes.  Returns `None` for
/// error.
pub fn imesh_dup_marked_conts(obj: &mut Iobj, flag: u32) -> Option<Iobj> {
    let mut err = 0;
    let mut maxsurf = 0;

    let Some(mut new_obj) = imod_object_new() else {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("Error duplicating contours for meshing\n"),
        );
        return None;
    };

    /* Copy object structure but zero out the count of mesh and contours in case
    we have to free it */
    imod_object_copy(obj, &mut new_obj);
    new_obj.cont = Vec::new();
    new_obj.store = Vec::new();
    new_obj.label = None;
    new_obj.mesh_param = None;
    new_obj.mesh = Vec::new();

    /* Duplicate contours one at a time and add to object */

    for i in 0..obj.cont.len() {
        /* Skip if there is a flag and this contour is not marked; otherwise
        reset the flag before the copy.  Then skip if empty */
        if flag != 0 && (flag & obj.cont[i].flags) == 0 {
            continue;
        }
        if flag != 0 {
            obj.cont[i].flags &= !flag;
        }
        if obj.cont[i].pts.is_empty() {
            continue;
        }

        /* Duplicate the contour and all its data and add to new object */
        err = 1;
        let Some(cont) = imod_contour_dup(&obj.cont[i]) else {
            break;
        };

        maxsurf = maxsurf.max(cont.surf);
        let new_ind = imod_object_add_contour(&mut new_obj, cont);
        if new_ind < 0 {
            break;
        }

        err = istore_copy_cont_surf_items(&obj.store, &mut new_obj.store, i as i32, new_ind, 0);
        if err != 0 {
            break;
        }
    }

    /* Copy non-index items and surface items up to the maximum surface copied */
    if err == 0 {
        err = istore_copy_non_index(&obj.store, &mut new_obj.store);
    }
    if err == 0 {
        for i in 0..=maxsurf {
            err = istore_copy_cont_surf_items(&obj.store, &mut new_obj.store, i, i, 1);
            if err != 0 {
                break;
            }
        }
    }

    if err != 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("Error duplicating contours for meshing\n"),
        );
        return None;
    }

    Some(new_obj)
}
