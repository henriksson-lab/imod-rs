//! Translation of `IMOD/libimod/iplane.c` -- "graphic plane library elements",
//! together with the clip-plane declarations of `IMOD/include/imodel.h` that it
//! operates on (`Iplane` and `IclipPlanes` live in `imodel.rs` with the rest of
//! that header).
//!
//! Deviation note: `imodClipsRead` (`iplane.c:165`) reads through
//! `imodGetBytes`/`imodGetFloats` (`imodel_files.c:1961`, `:2000`), which the
//! translated `imodel_files` unit does not carry as separate functions; as in
//! `imodel_read_header` there, the byte and float reads are done in place with
//! `read_exact`/`imod_get_float`.
#![allow(unused_variables)]

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libimod::imat::{
    Imat, imod_mat_delete, imod_mat_new, imod_mat_rot, imod_mat_rotate_vector, imod_mat_transform,
};
use crate::imod::libimod::imodel::{
    IMOD_CLIPSIZE, IMODF_MULTIPLE_CLIP, Iclip_planes, Iplane, Ipoint, SIZE_CLIP,
};
use crate::imod::libimod::imodel_files::imod_get_float;
use crate::imod::libimod::ipoint::imod_point_normalize;
use std::io::Read;

/// Original: `imodPlanesNew` (`iplane.c:20`).
///
/// Allocates an array of `size` `Iplane` structures, initializes them, and
/// returns the array, or `None` for error.
pub fn imod_planes_new(size: i32) -> Option<Vec<Iplane>> {
    if size <= 0 {
        return None;
    }

    let mut pl: Vec<Iplane> = vec![Iplane::default(); size as usize];
    for p in 0..size as usize {
        imod_plane_init(&mut pl[p]);
    }
    Some(pl)
}

/// Original: `imodPlaneDelete` (`iplane.c:37`).
///
/// Frees an array of `Iplane` structures in `plane`.
pub fn imod_plane_delete(plane: &mut Vec<Iplane>) {
    plane.clear();
}

/// Original: `imodPlaneInit` (`iplane.c:46`).
///
/// Initializes `plane` to the X/Y plane with a negative Z normal.
pub fn imod_plane_init(plane: &mut Iplane) {
    /* default plane z <= 0 */
    plane.a = 0.0f32;
    plane.b = 0.0f32;
    plane.d = 0.0f32;
    plane.c = -1.0f32;
}

/// Original: `imodPlaneAxisRotate` (`iplane.c:52`).
///
/// Unused, does not maintain a fixed point.
pub fn imod_plane_axis_rotate(plane: &mut Iplane, angle: f64, axis: i32) {
    let mut rpt = Ipoint::default();
    let Some(mut mat) = imod_mat_new(3) else {
        return;
    };

    imod_mat_rot(&mut mat, angle, axis);
    /* The source casts `Iplane *` to `Ipoint *`, so a, b, c are x, y, z. */
    let pl = Ipoint {
        x: plane.a,
        y: plane.b,
        z: plane.c,
    };
    imod_mat_transform(&mat, &pl, &mut rpt);
    imod_mat_delete(&mut mat);
    plane.a = rpt.x;
    plane.b = rpt.y;
    plane.c = rpt.z;
}

/// Original: `imodPlaneRotate` (`iplane.c:66`).
///
/// Rotate plane by data in point x = alpha, y = beta, z = gamma.
pub fn imod_plane_rotate(plane: &mut Iplane, angle: f64, pnt: &Ipoint) {
    let mut rpt = Ipoint::default();
    let Some(mut mat) = imod_mat_new(3) else {
        return;
    };

    imod_mat_rotate_vector(&mut mat, angle, pnt);
    /* The source casts `Iplane *` to `Ipoint *`, so a, b, c are x, y, z. */
    let pl = Ipoint {
        x: plane.a,
        y: plane.b,
        z: plane.c,
    };
    imod_mat_transform(&mat, &pl, &mut rpt);
    imod_mat_delete(&mut mat);
    plane.a = rpt.x;
    plane.b = rpt.y;
    plane.c = rpt.z;
}

/// Original: `imodPlaneSetPN` (`iplane.c:84`).
///
/// Set parameters of `plane` from a point `pnt` on a plane and a normal
/// vector `nor`.
pub fn imod_plane_set_pn(plane: &mut Iplane, pnt: &Ipoint, nor: &Ipoint) {
    plane.a = nor.x;
    plane.b = nor.y;
    plane.c = nor.z;
    plane.d = (pnt.x * nor.x) + (pnt.y * nor.y) + (pnt.z * nor.z);
}

/// Original: `imodPlaneClip` (`iplane.c:97`).
///
/// Tests whether point `pnt` is in the half space defined by the clipping
/// plane `plane`; returns 1 if it is or 0 if the point needs to be clipped.
pub fn imod_plane_clip(plane: &Iplane, pnt: &Ipoint) -> i32 {
    if ((plane.a * pnt.x) + (plane.b * pnt.y) + (plane.c * pnt.z) + plane.d) >= 0. {
        1
    } else {
        0
    }
}

/// Original: `imodPlanesClip` (`iplane.c:110`).
///
/// Tests whether point `pnt` is in the region defined by the `nplanes`
/// clipping planes in `plane`.
pub fn imod_planes_clip(plane: &[Iplane], nplanes: i32, pnt: &Ipoint) -> i32 {
    for pn in 0..nplanes as usize {
        if imod_plane_clip(&plane[pn], pnt) == 0 {
            return 0;
        }
    }
    1
}

/// Original: `imodClipsInitialize` (`iplane.c:124`).
///
/// Initializes all clip planes and other parameters of the clip plane set in
/// `clips`.
pub fn imod_clips_initialize(clips: &mut Iclip_planes) {
    clips.count = 0;
    clips.flags = 0;
    clips.trans = 0;
    clips.plane = 0;
    for i in 0..IMOD_CLIPSIZE {
        clips.normal[i].x = 0.0f32;
        clips.normal[i].y = 0.0f32;
        clips.normal[i].z = -1.0f32;
        clips.point[i].x = 0.0f32;
        clips.point[i].y = 0.0f32;
        clips.point[i].z = 0.0f32;
    }
}

/// Original: `imodClipsCopy` (`iplane.c:141`).
///
/// Copies the clip planes and parameters of the clip plane set in
/// `from_clips` to `to_clips`.
pub fn imod_clips_copy(from_clips: &Iclip_planes, to_clips: &mut Iclip_planes) {
    imod_clips_initialize(to_clips);
    to_clips.count = from_clips.count;
    to_clips.flags = from_clips.flags;
    to_clips.trans = from_clips.trans;
    to_clips.plane = from_clips.plane;
    for i in 0..from_clips.count as usize {
        to_clips.normal[i] = from_clips.normal[i];
        to_clips.point[i] = from_clips.point[i];
    }
}

/// Original: `imodClipsRead` (`iplane.c:160`).
///
/// Reads all the clip planes of a set into `clips` from the model file in
/// `fin`; the number of vectors and normals read is based on the size of the
/// data chunk.  Returns non-zero for a read error, standing in for the
/// source's `ferror(fin)`.
pub fn imod_clips_read(clips: &mut Iclip_planes, fin: &mut ImodFile) -> i32 {
    let mut size_bytes = [0_u8; 4];
    if fin.read_exact(&mut size_bytes).is_err() {
        return 1;
    }
    let size = i32::from_be_bytes(size_bytes);
    let nread = (size - SIZE_CLIP) / 24 + 1;

    /* imodGetBytes(fin, (unsigned char *)&clips->count, 4) -- the four bytes
    are count, flags, trans and plane in structure order. */
    let mut head = [0_u8; 4];
    if fin.read_exact(&mut head).is_err() {
        return 1;
    }
    clips.count = head[0];
    clips.flags = head[1];
    clips.trans = head[2];
    clips.plane = head[3];

    /* imodGetFloats(fin, (float *)&clips->normal[0], 3 * nread) */
    for i in 0..nread {
        for c in 0..3 {
            let Ok(v) = imod_get_float(fin) else {
                return 1;
            };
            if (i as usize) < clips.normal.len() {
                match c {
                    0 => clips.normal[i as usize].x = v,
                    1 => clips.normal[i as usize].y = v,
                    _ => clips.normal[i as usize].z = v,
                }
            }
        }
    }
    /* imodGetFloats(fin, (float *)&clips->point[0], 3 * nread) */
    for i in 0..nread {
        for c in 0..3 {
            let Ok(v) = imod_get_float(fin) else {
                return 1;
            };
            if (i as usize) < clips.point.len() {
                match c {
                    0 => clips.point[i as usize].x = v,
                    1 => clips.point[i as usize].y = v,
                    _ => clips.point[i as usize].z = v,
                }
            }
        }
    }
    0
}

/// Original: `imodClipsFixCount` (`iplane.c:186`).
///
/// Fixes the count of clip planes in `clips` for old files or for a count set
/// to zero when written in a new file.  `flags` should be the flags from the
/// model structure.
pub fn imod_clips_fix_count(clips: &mut Iclip_planes, flags: u32) {
    if clips.count == 0
        && (clips.point[0].x != 0.
            || clips.point[0].y != 0.
            || clips.point[0].z != 0.
            || clips.normal[0].x != 0.
            || clips.normal[0].y != 0.
            || clips.normal[0].z != -1.0f32)
    {
        clips.flags &= 254;
        clips.count = 1;
    } else if clips.count != 0 && (flags & IMODF_MULTIPLE_CLIP) == 0 {
        clips.flags |= 1;
        clips.count = 1;
    }
}

/// Original: `imodPlaneSetFromClips` (`iplane.c:210`).
///
/// Sets plane parameters in the `plane` array from the clipping plane sets in
/// both `obj_clips` and `glb_clips`, based upon which planes are on in each
/// set.
pub fn imod_plane_set_from_clips(
    obj_clips: Option<&Iclip_planes>,
    glb_clips: Option<&Iclip_planes>,
    plane: &mut [Iplane],
    max_planes: i32,
    n_planes: &mut i32,
) {
    let mut do_global = 1;
    if let Some(obj_clips) = obj_clips {
        if obj_clips.flags & (1 << 7) != 0 {
            do_global = 0;
        }
        for i in 0..obj_clips.count as usize {
            if (obj_clips.flags & (1 << i)) != 0 && *n_planes < max_planes {
                let np = *n_planes as usize;
                imod_plane_set_pn(&mut plane[np], &obj_clips.point[i], &obj_clips.normal[i]);
                *n_planes += 1;
            }
        }
    }

    if do_global != 0 {
        if let Some(glb_clips) = glb_clips {
            for i in 0..glb_clips.count as usize {
                if (glb_clips.flags & (1 << i)) != 0 && *n_planes < max_planes {
                    let np = *n_planes as usize;
                    imod_plane_set_pn(&mut plane[np], &glb_clips.point[i], &glb_clips.normal[i]);
                    *n_planes += 1;
                }
            }
        }
    }
}

/// Original: `imodClipsTrans` (`iplane.c:238`).
///
/// Transforms the clipping planes in `clips`, using `mat` to transform the
/// points and `mat2` to transform the normals.
pub fn imod_clips_trans(clips: &mut Iclip_planes, mat: &Imat, mat2: &Imat) {
    let mut pnt = Ipoint::default();
    let mut pnt2 = Ipoint::default();
    for i in 0..clips.count as usize {
        /* The clipping point is maintained as the negative of an actual
        location so it needs to be inverted, transformed, then reinverted */
        pnt2.x = -clips.point[i].x;
        pnt2.y = -clips.point[i].y;
        pnt2.z = -clips.point[i].z;
        imod_mat_transform(mat, &pnt2, &mut pnt);
        clips.point[i].x = -pnt.x;
        clips.point[i].y = -pnt.y;
        clips.point[i].z = -pnt.z;

        /* Transform and renormalize the normal */
        imod_mat_transform(mat2, &clips.normal[i], &mut pnt);
        imod_point_normalize(&mut pnt);
        clips.normal[i] = pnt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imat::{B3D_X, B3D_Y, B3D_Z, imod_mat_rot, imod_mat_scale};

    /// `%.9g` through the C library, so the driver's formatting is part of the
    /// comparison.
    fn g9(v: f64) -> String {
        crate::imod::libcfshr::b3dutil::c_format(
            "%.9g",
            &[crate::imod::libcfshr::b3dutil::CArg::Dbl(v)],
        )
    }

    /// Differential harness against a driver compiled directly against the
    /// pinned `IMOD/libimod/iplane.c` and linked to the reference build's
    /// `libimod`; the expected text below is that driver's verbatim output.
    #[test]
    fn source_c_driver_differential() {
        let mut out = String::new();
        out.push_str("--- iplane ---\n");

        let mut pl = imod_planes_new(3).unwrap();
        for i in 0..3 {
            out.push_str(&format!(
                "pl{i} {} {} {} {}\n",
                g9(pl[i].a as f64),
                g9(pl[i].b as f64),
                g9(pl[i].c as f64),
                g9(pl[i].d as f64)
            ));
        }
        let a = Ipoint {
            x: 1.,
            y: 2.,
            z: 3.,
        };
        let b = Ipoint {
            x: 0.3,
            y: -0.4,
            z: 0.86602540,
        };
        imod_plane_set_pn(&mut pl[0], &a, &b);
        out.push_str(&format!(
            "setpn {} {} {} {}\n",
            g9(pl[0].a as f64),
            g9(pl[0].b as f64),
            g9(pl[0].c as f64),
            g9(pl[0].d as f64)
        ));
        let c = Ipoint {
            x: 5.,
            y: -1.,
            z: 2.5,
        };
        out.push_str(&format!(
            "clip {} {}\n",
            imod_plane_clip(&pl[0], &c),
            imod_planes_clip(&pl, 3, &c)
        ));
        let d = Ipoint {
            x: -20.,
            y: 7.,
            z: -3.,
        };
        out.push_str(&format!(
            "clip2 {} {}\n",
            imod_plane_clip(&pl[0], &d),
            imod_planes_clip(&pl, 3, &d)
        ));

        imod_plane_rotate(&mut pl[1], 37.5, &b);
        out.push_str(&format!(
            "prot {} {} {} {}\n",
            g9(pl[1].a as f64),
            g9(pl[1].b as f64),
            g9(pl[1].c as f64),
            g9(pl[1].d as f64)
        ));
        imod_plane_axis_rotate(&mut pl[2], 22.5, B3D_Y);
        out.push_str(&format!(
            "parot {} {} {} {}\n",
            g9(pl[2].a as f64),
            g9(pl[2].b as f64),
            g9(pl[2].c as f64),
            g9(pl[2].d as f64)
        ));
        imod_plane_delete(&mut pl);

        let mut clips = Iclip_planes::default();
        imod_clips_initialize(&mut clips);
        out.push_str(&format!(
            "cinit {} {} {} {}\n",
            clips.count, clips.flags, clips.trans, clips.plane
        ));
        for i in 0..IMOD_CLIPSIZE {
            out.push_str(&format!(
                "cn{i} {} {} {} {} {} {}\n",
                g9(clips.normal[i].x as f64),
                g9(clips.normal[i].y as f64),
                g9(clips.normal[i].z as f64),
                g9(clips.point[i].x as f64),
                g9(clips.point[i].y as f64),
                g9(clips.point[i].z as f64)
            ));
        }
        clips.count = 3;
        clips.flags = 5;
        clips.trans = 9;
        clips.plane = 2;
        for i in 0..3 {
            clips.normal[i].x = 0.1 * (i + 1) as f32;
            clips.normal[i].y = -0.5 * (i + 1) as f32;
            clips.normal[i].z = 1. - 0.25 * i as f32;
            clips.point[i].x = -3. * (i + 1) as f32;
            clips.point[i].y = 4. + i as f32;
            clips.point[i].z = 0.5 * i as f32;
        }
        let mut clips2 = Iclip_planes::default();
        imod_clips_copy(&clips, &mut clips2);
        out.push_str(&format!(
            "ccopy {} {} {} {}\n",
            clips2.count, clips2.flags, clips2.trans, clips2.plane
        ));
        for i in 0..IMOD_CLIPSIZE {
            out.push_str(&format!(
                "cc{i} {} {} {} {} {} {}\n",
                g9(clips2.normal[i].x as f64),
                g9(clips2.normal[i].y as f64),
                g9(clips2.normal[i].z as f64),
                g9(clips2.point[i].x as f64),
                g9(clips2.point[i].y as f64),
                g9(clips2.point[i].z as f64)
            ));
        }

        let mut planes = [Iplane::default(); 10];
        let mut nplanes = 0;
        imod_plane_set_from_clips(Some(&clips), Some(&clips2), &mut planes, 10, &mut nplanes);
        out.push_str(&format!("setfrom {nplanes}\n"));
        for i in 0..nplanes as usize {
            out.push_str(&format!(
                "sf{i} {} {} {} {}\n",
                g9(planes[i].a as f64),
                g9(planes[i].b as f64),
                g9(planes[i].c as f64),
                g9(planes[i].d as f64)
            ));
        }
        nplanes = 0;
        imod_plane_set_from_clips(None, Some(&clips2), &mut planes, 2, &mut nplanes);
        out.push_str(&format!("setfrom2 {nplanes}\n"));
        for i in 0..nplanes as usize {
            out.push_str(&format!(
                "sg{i} {} {} {} {}\n",
                g9(planes[i].a as f64),
                g9(planes[i].b as f64),
                g9(planes[i].c as f64),
                g9(planes[i].d as f64)
            ));
        }
        clips2.flags |= 1 << 7;
        nplanes = 0;
        imod_plane_set_from_clips(Some(&clips2), Some(&clips), &mut planes, 10, &mut nplanes);
        out.push_str(&format!("setfrom3 {nplanes}\n"));
        for i in 0..nplanes as usize {
            out.push_str(&format!(
                "sh{i} {} {} {} {}\n",
                g9(planes[i].a as f64),
                g9(planes[i].b as f64),
                g9(planes[i].c as f64),
                g9(planes[i].d as f64)
            ));
        }

        let mut mat = imod_mat_new(3).unwrap();
        imod_mat_rot(&mut mat, 30., B3D_Z);
        imod_mat_rot(&mut mat, -17., B3D_X);
        let s = Ipoint {
            x: 1.5,
            y: 0.75,
            z: 2.,
        };
        imod_mat_scale(&mut mat, &s);
        let mut mat2 = imod_mat_new(3).unwrap();
        imod_mat_rot(&mut mat2, 30., B3D_Z);
        imod_clips_trans(&mut clips, &mat, &mat2);
        out.push_str("ctrans\n");
        for i in 0..IMOD_CLIPSIZE {
            out.push_str(&format!(
                "ct{i} {} {} {} {} {} {}\n",
                g9(clips.normal[i].x as f64),
                g9(clips.normal[i].y as f64),
                g9(clips.normal[i].z as f64),
                g9(clips.point[i].x as f64),
                g9(clips.point[i].y as f64),
                g9(clips.point[i].z as f64)
            ));
        }

        let mut fc = Iclip_planes::default();
        imod_clips_initialize(&mut fc);
        imod_clips_fix_count(&mut fc, 0);
        out.push_str(&format!("fix0 {} {}\n", fc.count, fc.flags));
        imod_clips_initialize(&mut fc);
        fc.point[0].x = 3.;
        imod_clips_fix_count(&mut fc, 0);
        out.push_str(&format!("fix1 {} {}\n", fc.count, fc.flags));
        imod_clips_initialize(&mut fc);
        fc.count = 4;
        fc.flags = 8;
        imod_clips_fix_count(&mut fc, 0);
        out.push_str(&format!("fix2 {} {}\n", fc.count, fc.flags));
        imod_clips_initialize(&mut fc);
        fc.count = 4;
        fc.flags = 8;
        imod_clips_fix_count(&mut fc, IMODF_MULTIPLE_CLIP);
        out.push_str(&format!("fix3 {} {}\n", fc.count, fc.flags));

        let want = EXPECTED_C_DRIVER_OUTPUT;
        for (line, (got, want)) in out.lines().zip(want.lines()).enumerate() {
            assert_eq!(got, want, "line {} differs from iplane.c driver", line + 1);
        }
        assert_eq!(out.lines().count(), want.lines().count());
        let _ = B3D_X;
    }

    /// Verbatim stdout of the driver linked against `IMOD/libimod/iplane.c`.
    const EXPECTED_C_DRIVER_OUTPUT: &str = r#"--- iplane ---
pl0 0 0 -1 0
pl1 0 0 -1 0
pl2 0 0 -1 0
setpn 0.300000012 -0.400000006 0.866025388 2.09807611
clip 1 0
clip2 0 0
prot 0.189816207 0.254212946 -0.94833833 0
parot -0.382683456 0 -0.923879504 0
cinit 0 0 0 0
cn0 0 0 -1 0 0 0
cn1 0 0 -1 0 0 0
cn2 0 0 -1 0 0 0
cn3 0 0 -1 0 0 0
cn4 0 0 -1 0 0 0
cn5 0 0 -1 0 0 0
ccopy 3 5 9 2
cc0 0.100000001 -0.5 1 -3 4 0
cc1 0.200000003 -1 0.75 -6 5 0.5
cc2 0.300000012 -1.5 0.5 -9 6 1
cc3 0 0 -1 0 0 0
cc4 0 0 -1 0 0 0
cc5 0 0 -1 0 0 0
setfrom 4
sf0 0.100000001 -0.5 1 -2.29999995
sf1 0.300000012 -1.5 0.5 -11.1999998
sf2 0.100000001 -0.5 1 -2.29999995
sf3 0.300000012 -1.5 0.5 -11.1999998
setfrom2 2
sg0 0.100000001 -0.5 1 -2.29999995
sg1 0.300000012 -1.5 0.5 -11.1999998
setfrom3 2
sh0 0.100000001 -0.5 1 -2.29999995
sh1 0.300000012 -1.5 0.5 -11.1999998
ctrans
ct0 0.299869388 -0.341214806 0.89087081 -6.89711428 1.40870976 -1.14849555
ct1 0.531800091 -0.605123699 0.592464447 -11.5442286 1.06364441 0.178521574
ct2 0.627463937 -0.713977516 0.31068489 -16.1913414 0.718579531 1.50553906
ct3 0 0 -1 0 0 0
ct4 0 0 -1 0 0 0
ct5 0 0 -1 0 0 0
fix0 0 0
fix1 1 0
fix2 1 9
fix3 4 8
"#;
}
