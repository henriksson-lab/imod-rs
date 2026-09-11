//! Export routines from `IMOD/libimod/imodel_to.c`.

use std::io::Write;

use super::imat::{imod_mat_delete, imod_mat_new, imod_mat_rot, imod_mat_transform};
use super::imodel::{
    IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2, IMOD_MESH_END, IMOD_MESH_ENDPOLY,
    IMOD_OBJFLAG_OPEN, Imesh, Imod, Iobj, Ipoint,
};
use super::iobj::{
    IOBJ_FLAG_MESH, imod_object_get_value, imod_object_sort, iobj_close, iobj_off, iobj_open,
    iobj_scat,
};
use super::iview::imod_view_model_default;

/// Original: `imod_to_wmod` (`imodel_to.c:36`).
pub fn imod_to_wmod(imod: &Imod, output: &mut impl Write, filename: &str) -> Result<(), i32> {
    let contour_count: usize = imod.obj.iter().map(|object| object.cont.len()).sum();
    let point_total: usize = imod
        .obj
        .iter()
        .flat_map(|object| &object.cont)
        .map(|contour| contour.pts.len())
        .sum();
    write!(output, " Model file name........................{filename}").map_err(|_| 11)?;
    let blank_count = 31_i32 - filename.len() as i32;
    if blank_count < 0 {
        for _ in 0..blank_count {
            write!(output, " ").map_err(|_| 11)?;
        }
    }
    writeln!(output).map_err(|_| 11)?;
    writeln!(
        output,
        " max # of object....................... {:4}",
        2 * contour_count
    )
    .map_err(|_| 11)?;
    writeln!(
        output,
        " # of node............................. {:4}",
        2 * point_total
    )
    .map_err(|_| 11)?;
    writeln!(
        output,
        " # of object........................... {:4}",
        contour_count
    )
    .map_err(|_| 11)?;
    writeln!(output, "  Object sequence : ").map_err(|_| 11)?;
    let mut object_count = 1_i32;
    let mut point_count = 0_i32;
    let mut display_switch = 247_i32;
    for object in &imod.obj {
        if display_switch > 255 {
            display_switch = 247;
        }
        for contour in &object.cont {
            writeln!(output, "  Object #: {:11}", object_count).map_err(|_| 11)?;
            object_count += 1;
            writeln!(output, " # of point: {:11}", contour.pts.len()).map_err(|_| 11)?;
            writeln!(output, " Display switch:1  {display_switch}").map_err(|_| 11)?;
            writeln!(output, "     #    X       Y       Z      Mark    Label ").map_err(|_| 11)?;
            point_count += 1;
            for point in &contour.pts {
                write!(output, "{:7}", point_count).map_err(|_| 11)?;
                point_count += 1;
                write!(output, " {:7.2}", point.x).map_err(|_| 11)?;
                write!(output, " {:7.2}", point.y).map_err(|_| 11)?;
                writeln!(output, " {:7.2}   0", point.z).map_err(|_| 11)?;
            }
        }
    }
    writeln!(output, "\n  END").map_err(|_| 11)
}

/// Original: `imod_to_nff` (`imodel_to.c:108`).
///
/// `imodMeshPolyNormFactors` (`imesh.c:312`) has no translated module yet, so
/// its three index factors are computed in place.
pub fn imod_to_nff(mod_: &Imod, fout: *mut libc::FILE) -> i32 {
    let mut _xo = 0f32;
    let _yo = 0f32;
    let _zo = 0f32; /* x,y,z offsets. */
    let _xs = 1f32;
    let _ys = 1f32;
    let _zs = 1f32; /* x,y,z scale.   */

    _xo = mod_.xoffset;

    /* Loop through objects. */
    for objnum in 0..mod_.obj.len() {
        /* Set color for each object. */
        let obj = &mod_.obj[objnum];

        unsafe {
            libc::fprintf(
                fout,
                c"# object with %d contours.\n".as_ptr(),
                obj.cont.len() as std::ffi::c_int,
            );
            libc::fprintf(
                fout,
                c"f %g %g %g 0 0 0 0 0\n".as_ptr(),
                obj.red as std::ffi::c_double,
                obj.green as std::ffi::c_double,
                obj.blue as std::ffi::c_double,
            );
        }

        let use_mesh = imod_object_get_value(obj, IOBJ_FLAG_MESH);

        if use_mesh != 0 {
            for mi in 0..obj.mesh.len() {
                let mesh = &obj.mesh[mi];

                let mut i = 0usize;
                while i < mesh.list.len() {
                    match mesh.list[i] {
                        /* IMOD_MESH_BGNPOLY, IMOD_MESH_BGNBIGPOLY */
                        -21 | -24 => {
                            i += 1;
                            while mesh.list[i] != IMOD_MESH_ENDPOLY {
                                i += 1;
                            }
                        }

                        IMOD_MESH_BGNPOLYNORM | IMOD_MESH_BGNPOLYNORM2 => {
                            /* imodMeshPolyNormFactors (`imesh.c:312`) */
                            let (list_inc, vert_base, norm_add) =
                                if mesh.list[i] == IMOD_MESH_BGNPOLYNORM {
                                    (2usize, 1usize, 0i32)
                                } else {
                                    (1usize, 0usize, 1i32)
                                };
                            i += 1;
                            while mesh.list[i] != IMOD_MESH_ENDPOLY {
                                unsafe { libc::fprintf(fout, c"pp 3\n".as_ptr()) };
                                for _ in 0..3 {
                                    let vert = mesh.vert[mesh.list[i + vert_base] as usize];
                                    let norm = mesh.vert[(mesh.list[i] + norm_add) as usize];
                                    unsafe {
                                        libc::fprintf(
                                            fout,
                                            c"%g %g %g %g %g %g\n".as_ptr(),
                                            (vert.x * mod_.xscale) as std::ffi::c_double,
                                            (vert.y * mod_.yscale) as std::ffi::c_double,
                                            (vert.z * mod_.zscale) as std::ffi::c_double,
                                            (norm.x * mod_.xscale) as std::ffi::c_double,
                                            (norm.y * mod_.yscale) as std::ffi::c_double,
                                            (norm.z * mod_.zscale) as std::ffi::c_double,
                                        );
                                    }
                                    i += list_inc;
                                }
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
        } else {
            /* Loop through contours. */
            for contnum in 0..obj.cont.len() {
                let cont = &obj.cont[contnum];

                /* Loop through points. */

                if iobj_scat(obj.flags) != 0 {
                    for i in 0..cont.pts.len() {
                        unsafe {
                            libc::fprintf(
                                fout,
                                c"s %g %g %g %d\n".as_ptr(),
                                ((cont.pts[i].x + mod_.xoffset) * mod_.xscale)
                                    as std::ffi::c_double,
                                ((cont.pts[i].y + mod_.yoffset) * mod_.yscale)
                                    as std::ffi::c_double,
                                ((cont.pts[i].z + mod_.zoffset) * mod_.zscale)
                                    as std::ffi::c_double,
                                obj.pdrawsize as std::ffi::c_int,
                            );
                        }
                    }
                } else if (obj.flags & IMOD_OBJFLAG_OPEN) == 0 {
                    unsafe {
                        libc::fprintf(fout, c"p %d\n".as_ptr(), cont.pts.len() as std::ffi::c_int)
                    };
                    for i in 0..cont.pts.len() {
                        unsafe {
                            libc::fprintf(
                                fout,
                                c"%g %g %g\n".as_ptr(),
                                ((cont.pts[i].x + mod_.xoffset) * mod_.xscale)
                                    as std::ffi::c_double,
                                ((cont.pts[i].y + mod_.yoffset) * mod_.yscale)
                                    as std::ffi::c_double,
                                ((cont.pts[i].z + mod_.zoffset) * mod_.zscale)
                                    as std::ffi::c_double,
                            );
                        }
                    }
                } else {
                    if cont.pts.len() == 1 {
                        unsafe { libc::fprintf(fout, c"p %d\n".as_ptr(), 3 as std::ffi::c_int) };
                        for _ in 0..3 {
                            unsafe {
                                libc::fprintf(
                                    fout,
                                    c"%g %g %g\n".as_ptr(),
                                    ((cont.pts[0].x + mod_.xoffset) * mod_.xscale)
                                        as std::ffi::c_double,
                                    ((cont.pts[0].y + mod_.yoffset) * mod_.yscale)
                                        as std::ffi::c_double,
                                    ((cont.pts[0].z + mod_.zoffset) * mod_.zscale)
                                        as std::ffi::c_double,
                                );
                            }
                        }
                    }

                    if cont.pts.len() == 2 {
                        unsafe {
                            libc::fprintf(fout, c"p %d\n".as_ptr(), 3 as std::ffi::c_int);
                            libc::fprintf(
                                fout,
                                c"%g %g %g\n".as_ptr(),
                                ((cont.pts[0].x + mod_.xoffset) * mod_.xscale)
                                    as std::ffi::c_double,
                                ((cont.pts[0].y + mod_.yoffset) * mod_.yscale)
                                    as std::ffi::c_double,
                                ((cont.pts[0].z + mod_.zoffset) * mod_.zscale)
                                    as std::ffi::c_double,
                            );

                            libc::fprintf(
                                fout,
                                c"%g %g %g\n".as_ptr(),
                                ((cont.pts[1].x + mod_.xoffset) * mod_.xscale)
                                    as std::ffi::c_double,
                                ((cont.pts[1].y + mod_.yoffset) * mod_.yscale)
                                    as std::ffi::c_double,
                                ((cont.pts[1].z + mod_.zoffset) * mod_.zscale)
                                    as std::ffi::c_double,
                            );

                            libc::fprintf(
                                fout,
                                c"%g %g %g\n".as_ptr(),
                                ((cont.pts[0].x + mod_.xoffset) * mod_.xscale)
                                    as std::ffi::c_double,
                                ((cont.pts[0].y + mod_.yoffset) * mod_.yscale)
                                    as std::ffi::c_double,
                                ((cont.pts[0].z + mod_.zoffset) * mod_.zscale)
                                    as std::ffi::c_double,
                            );
                        }
                    }

                    if cont.pts.len() > 2 {
                        unsafe {
                            libc::fprintf(
                                fout,
                                c"p %d\n".as_ptr(),
                                ((cont.pts.len() * 2) - 2) as std::ffi::c_int,
                            )
                        };
                        for i in 0..cont.pts.len() {
                            unsafe {
                                libc::fprintf(
                                    fout,
                                    c"%g %g %g\n".as_ptr(),
                                    ((cont.pts[i].x + mod_.xoffset) * mod_.xscale)
                                        as std::ffi::c_double,
                                    ((cont.pts[i].y + mod_.yoffset) * mod_.yscale)
                                        as std::ffi::c_double,
                                    ((cont.pts[i].z + mod_.zoffset) * mod_.zscale)
                                        as std::ffi::c_double,
                                );
                            }
                        }

                        let mut i = cont.pts.len() as i32 - 2;
                        while i > 0 {
                            unsafe {
                                libc::fprintf(
                                    fout,
                                    c"%g %g %g\n".as_ptr(),
                                    ((cont.pts[i as usize].x + mod_.xoffset) * mod_.xscale)
                                        as std::ffi::c_double,
                                    ((cont.pts[i as usize].y + mod_.yoffset) * mod_.yscale)
                                        as std::ffi::c_double,
                                    ((cont.pts[i as usize].z + mod_.zoffset) * mod_.zscale)
                                        as std::ffi::c_double,
                                );
                            }
                            i -= 1;
                        }
                    }
                }
            }
        }
    }
    0
}

/// Original: `imod_to_synu` (`imodel_to.c:255`).
///
/// Writes `type<n>.cont`, `type<n>.mesh` and `Viewdata` into the working
/// directory exactly as the source does.
pub fn imod_to_synu(mod_: &mut Imod) -> i32 {
    let vdata = c"visible=no rep=l trans=no cull=off depthcue=yes shademode=f";
    let mdata = c"visible=no rep=l trans=yes cull=off depthcue=yes shademode=s";
    let viewdata = c"Viewdata";

    for ob in 0..mod_.obj.len() {
        imod_object_sort(&mut mod_.obj[ob]);
    }

    for ob in 0..mod_.obj.len() {
        let filename = std::ffi::CString::new(format!("type{}.cont", ob)).unwrap();
        let fout = unsafe { libc::fopen(filename.as_ptr(), c"w".as_ptr()) };
        if fout.is_null() {
            continue;
        }

        for co in 0..mod_.obj[ob].cont.len() {
            let cont = &mod_.obj[ob].cont[co];
            let mut vertices = cont.pts.len() as i32;
            let v = vertices;
            let mut edges = (2 * v) - 2;

            if vertices == 1 {
                vertices = 2;
            }
            if edges == 0 {
                edges = 2;
            }

            if vertices == 0 {
                continue;
            }

            unsafe {
                libc::fprintf(
                    fout,
                    c"#synu\n#Imod object %d\n".as_ptr(),
                    ob as std::ffi::c_int,
                );
            }
            if mod_.obj[ob].flags & IMOD_OBJFLAG_OPEN != 0 {
                unsafe {
                    libc::fprintf(
                        fout,
                        c"polygonmesh 4l %dl %dl 1l\n%d\n%d\n1\n0\n".as_ptr(),
                        vertices as std::ffi::c_int,
                        edges as std::ffi::c_int,
                        vertices as std::ffi::c_int,
                        edges as std::ffi::c_int,
                    );
                }

                if v > 1 {
                    for pt in 0..v as usize {
                        unsafe {
                            libc::fprintf(
                                fout,
                                c"%5.2f %5.2f %5.2f\n".as_ptr(),
                                cont.pts[pt].x as std::ffi::c_double,
                                cont.pts[pt].y as std::ffi::c_double,
                                (cont.pts[pt].z * mod_.zscale) as std::ffi::c_double,
                            );
                        }
                    }
                    for pt in 0..v {
                        unsafe { libc::fprintf(fout, c"%d\n".as_ptr(), pt as std::ffi::c_int) };
                    }
                    let mut pt = v - 2;
                    while pt > 0 {
                        unsafe { libc::fprintf(fout, c"%d\n".as_ptr(), pt as std::ffi::c_int) };
                        pt -= 1;
                    }
                    unsafe {
                        libc::fprintf(fout, c"%d\n".as_ptr(), ((2 * v) - 3) as std::ffi::c_int)
                    };
                } else {
                    unsafe {
                        libc::fprintf(
                            fout,
                            c"%5.2f %5.2f %5.2f\n".as_ptr(),
                            cont.pts[0].x as std::ffi::c_double,
                            cont.pts[0].y as std::ffi::c_double,
                            (cont.pts[0].z * mod_.zscale) as std::ffi::c_double,
                        );
                        libc::fprintf(
                            fout,
                            c"%5.2f %5.2f %5.2f\n".as_ptr(),
                            cont.pts[0].x as std::ffi::c_double,
                            cont.pts[0].y as std::ffi::c_double,
                            (cont.pts[0].z * mod_.zscale) as std::ffi::c_double,
                        );
                        libc::fprintf(fout, c"0\n1\n1\n".as_ptr());
                    }
                }
            } else {
                unsafe {
                    libc::fprintf(
                        fout,
                        c"polygonmesh 4l %dl %dl 1l\n%d\n%d\n1\n0\n".as_ptr(),
                        v as std::ffi::c_int,
                        v as std::ffi::c_int,
                        v as std::ffi::c_int,
                        v as std::ffi::c_int,
                    );
                }
                for pt in 0..v as usize {
                    unsafe {
                        libc::fprintf(
                            fout,
                            c"%5.2f %5.2f %5.2f\n".as_ptr(),
                            cont.pts[pt].x as std::ffi::c_double,
                            cont.pts[pt].y as std::ffi::c_double,
                            (cont.pts[pt].z * mod_.zscale) as std::ffi::c_double,
                        );
                    }
                }
                let mut pt = 0;
                while pt < v {
                    unsafe { libc::fprintf(fout, c"%d\n".as_ptr(), pt as std::ffi::c_int) };
                    pt += 1;
                }
                pt -= 1;
                unsafe { libc::fprintf(fout, c"%d\n".as_ptr(), pt as std::ffi::c_int) };
            }
        }

        unsafe { libc::fclose(fout) };

        if !mod_.obj[ob].mesh.is_empty() {
            let zscale = mod_.zscale as f64;
            imod_mesh_to_synu(&mod_.obj[ob], ob as i32, zscale);
        }
    }

    let fview = unsafe { libc::fopen(viewdata.as_ptr(), c"w".as_ptr()) };

    if !fview.is_null() {
        for ob in 0..mod_.obj.len() {
            unsafe {
                libc::fprintf(
                    fview,
                    c"type%d.cont %s ".as_ptr(),
                    ob as std::ffi::c_int,
                    vdata.as_ptr(),
                );
                libc::fprintf(
                    fview,
                    c"color=%1.2f,%1.2f,%1.2f,%1.2f\n".as_ptr(),
                    mod_.obj[ob].red as std::ffi::c_double,
                    mod_.obj[ob].green as std::ffi::c_double,
                    mod_.obj[ob].blue as std::ffi::c_double,
                    mod_.obj[ob].trans as std::ffi::c_double / 255.0,
                );
            }

            if !mod_.obj[ob].mesh.is_empty() {
                unsafe {
                    libc::fprintf(
                        fview,
                        c"type%d.mesh %s ".as_ptr(),
                        ob as std::ffi::c_int,
                        mdata.as_ptr(),
                    );
                    libc::fprintf(
                        fview,
                        c"color=%1.2f,%1.2f,%1.2f,%1.2f\n".as_ptr(),
                        mod_.obj[ob].red as std::ffi::c_double,
                        mod_.obj[ob].green as std::ffi::c_double,
                        mod_.obj[ob].blue as std::ffi::c_double,
                        mod_.obj[ob].trans as std::ffi::c_double / 255.0,
                    );
                }
            }
        }
        unsafe { libc::fclose(fview) };
    }

    0
}

/// Original: `imod_mesh_to_synu` (`imodel_to.c:381`).
pub fn imod_mesh_to_synu(obj: &Iobj, no: i32, zscale: f64) -> i32 {
    let filename = std::ffi::CString::new(format!("type{}.mesh", no)).unwrap();
    let fout = unsafe { libc::fopen(filename.as_ptr(), c"w".as_ptr()) };
    if fout.is_null() {
        return -1;
    }

    for i in 0..obj.mesh.len() {
        let mesh = &obj.mesh[i];
        if mesh.vert.is_empty() || mesh.list.is_empty() {
            continue;
        }
        let no_of_verts = mesh.vert.len() as i32;
        let no_of_polys = no_of_verts;
        let no_of_edges = no_of_polys * 3;

        unsafe {
            libc::fprintf(
                fout,
                c"#synu\n#Imod object %d\n#mesh %d\n".as_ptr(),
                no as std::ffi::c_int,
                i as std::ffi::c_int,
            );

            libc::fprintf(
                fout,
                c"polygonmesh 4l %dl %dl %dl\n%d\n%d\n%d\n0\n".as_ptr(),
                no_of_verts as std::ffi::c_int,
                no_of_edges as std::ffi::c_int,
                no_of_polys as std::ffi::c_int,
                no_of_verts as std::ffi::c_int,
                no_of_edges as std::ffi::c_int,
                no_of_polys as std::ffi::c_int,
            );
        }

        /* vertices list */
        for v in 0..no_of_verts as usize {
            unsafe {
                libc::fprintf(
                    fout,
                    c"%5.2f %5.2f %5.2f\n".as_ptr(),
                    mesh.vert[v].x as std::ffi::c_double,
                    mesh.vert[v].y as std::ffi::c_double,
                    mesh.vert[v].z as std::ffi::c_double * zscale,
                );
            }
        }

        /* edge list */

        let mut v1 = mesh.list[0];
        let mut v2 = mesh.list[1];
        let mut lowz = mesh.vert[0].z;
        let mut hiz = mesh.vert[mesh.vert.len() - 1].z;
        if lowz > hiz {
            lowz = hiz;
            hiz = mesh.vert[0].z;
        }

        let mut vc = 0i32;
        let mut e = 0i32;
        let mut l = 2usize;
        while e < no_of_edges {
            /* IMOD_MESH_SWAP (`imesh.h:22`) */
            if mesh.list[l] == -10 {
                let tv = v1;
                v1 = v2;
                v2 = tv;
            } else {
                e += 3;
                if (l as i32) < mesh.list.len() as i32 {
                    let v3 = mesh.list[l];
                    let mut direction = 1i32;
                    if mesh.vert[v1 as usize].z == mesh.vert[v2 as usize].z {
                        vc = v1 - v2;
                        if vc < 0 {
                            vc *= -1;
                        }
                        vc -= 1;
                        if mesh.vert[v1 as usize].z == lowz {
                            if v2 > v1 {
                                direction *= -1;
                            }
                        } else if v1 > v2 {
                            direction *= -1;
                        }
                    }
                    if mesh.vert[v2 as usize].z == mesh.vert[v3 as usize].z {
                        vc = v3 - v2;
                        if vc < 0 {
                            vc *= -1;
                        }
                        vc -= 1;
                        if mesh.vert[v1 as usize].z == hiz {
                            if v2 > v3 {
                                direction *= -1;
                            }
                        } else if v3 > v2 {
                            direction *= -1;
                        }
                    }
                    if mesh.vert[v3 as usize].z == mesh.vert[v1 as usize].z {
                        vc = v1 - v3;
                        if vc < 0 {
                            vc *= -1;
                        }
                        vc -= 1;
                        if mesh.vert[v1 as usize].z == hiz {
                            if v1 > v3 {
                                direction *= -1;
                            }
                        } else if v3 > v1 {
                            direction *= -1;
                        }
                    }

                    if vc != 0 {
                        direction *= -1;
                    }

                    if direction > 0 {
                        unsafe {
                            libc::fprintf(
                                fout,
                                c"%d\n%d\n%d\n".as_ptr(),
                                v3 as std::ffi::c_int,
                                v2 as std::ffi::c_int,
                                v1 as std::ffi::c_int,
                            )
                        };
                    } else {
                        unsafe {
                            libc::fprintf(
                                fout,
                                c"%d\n%d\n%d\n".as_ptr(),
                                v1 as std::ffi::c_int,
                                v2 as std::ffi::c_int,
                                v3 as std::ffi::c_int,
                            )
                        };
                    }
                    v1 = v2;
                    v2 = mesh.list[l];
                } else {
                    unsafe { libc::fprintf(fout, c"0\n1\n2\n".as_ptr()) };
                }
            }
            l += 1;
        }

        /* polygon list:  2, 5, 8, 11, ... */
        for p in 0..no_of_polys {
            unsafe { libc::fprintf(fout, c"%d\n".as_ptr(), ((p * 3) + 2) as std::ffi::c_int) };
        }
    }
    0
}

/* The Renderman (R) Interface Procedures and RIB Protocol are:
 * Copyright 1988,1989, Pixar.
 * All rights reseved.
 * RenderMan (R) is a registered trademark of Pixar.
 */

/// Original: `imod_to_RIB` (`imodel_to.c:523`).
pub fn imod_to_rib(imod: &mut Imod, fout: *mut libc::FILE) -> i32 {
    let image_max = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };

    let _image_filename = c"imod2RIB.tif";
    let mut opac: f32;
    let mut cscale: f32; /* camera scale */
    let mut fovytan: f32;
    let mut cdist: f32;

    let _xoffset: f32;
    let _yoffset: f32;
    let _zoffset: f32;
    let lxn = -0.5f32;
    let lyn = 0.5f32;
    let lzn = -1.0f32;
    let _fovy = 33.0f32; /* field of view of the camera, 0 means ortho */
    let _czoom = imod.xscale; /* camera zoom factor. */
    let r: f32; /* raduis of bounding sphere. */

    /* calculate camera scale */
    if imod.cview == 0 {
        let mut view = std::mem::take(&mut imod.view);
        imod_view_model_default(imod, &mut view[0], &image_max);
        imod.view = view;
    }
    let vw = &imod.view[imod.cview as usize];

    r = vw.rad;
    cscale = ((1.0f32 / r) as f64 * 0.75) as f32;
    fovytan = ((vw.fovy as f64) * 0.008_726_646_3).tan() as f32;
    if fovytan != 0. {
        cdist = r / fovytan;
    } else {
        cdist = r;
        fovytan = cscale;
    }
    let _ = fovytan;

    cdist = (cdist as f64 / 0.75) as f32;

    unsafe {
        libc::fprintf(fout, c"#RenderMan RIB-Structure 1.0\n".as_ptr());
        libc::fprintf(fout, c"#Created\n".as_ptr());
        libc::fprintf(fout, c"#by IMOD\n\n".as_ptr());

        libc::fprintf(
            fout,
            c"#The Renderman (R) Interface Procedures and RIB Protocol are:\n".as_ptr(),
        );
        libc::fprintf(fout, c"#Copyright 1988,1989, Pixar.\n".as_ptr());
        libc::fprintf(fout, c"#All rights reseved.\n".as_ptr());
        libc::fprintf(
            fout,
            c"#RenderMan (R) is a registered trademark of Pixar.\n".as_ptr(),
        );
        libc::fprintf(fout, c"#\n\n".as_ptr());

        if vw.fovy != 0. {
            libc::fprintf(
                fout,
                c"Projection \"perspective\"  \"fov\" %g\n".as_ptr(),
                vw.fovy as std::ffi::c_double,
            );
            cscale = 1.0;
        }

        /* Position light for Model */
        libc::fprintf(
            fout,
            c"LightSource \"distantlight\" 1 \"from\" [%g %g %g] ".as_ptr(),
            lxn as std::ffi::c_double,
            lyn as std::ffi::c_double,
            lzn as std::ffi::c_double,
        );
        libc::fprintf(fout, c"\"to\" [0 0 0] \"intensity\" 1\n".as_ptr());

        libc::fprintf(fout, c"\nWorldBegin\n".as_ptr());
        libc::fprintf(
            fout,
            c"Translate 0 0 %g\n".as_ptr(),
            cdist as std::ffi::c_double,
        );
        libc::fprintf(
            fout,
            c"Scale %g %g -%g\n".as_ptr(),
            cscale as std::ffi::c_double,
            cscale as std::ffi::c_double,
            cscale as std::ffi::c_double,
        );

        libc::fprintf(
            fout,
            c"Rotate %g 1 0 0\n".as_ptr(),
            vw.rot.x as std::ffi::c_double,
        );
        libc::fprintf(
            fout,
            c"Rotate %g 0 1 0\n".as_ptr(),
            vw.rot.y as std::ffi::c_double,
        );
        libc::fprintf(
            fout,
            c"Rotate %g 0 0 1\n".as_ptr(),
            vw.rot.z as std::ffi::c_double,
        );

        libc::fprintf(
            fout,
            c"Translate %g %g %g\n".as_ptr(),
            vw.trans.x as std::ffi::c_double,
            vw.trans.y as std::ffi::c_double,
            vw.trans.z as std::ffi::c_double,
        );
    }

    /* Draw each object */
    for ob in 0..imod.obj.len() {
        let obj = &imod.obj[ob];
        unsafe {
            libc::fprintf(
                fout,
                c"\n# Object %d\n".as_ptr(),
                (ob + 1) as std::ffi::c_int,
            )
        };
        if obj.name[0] != 0 {
            unsafe { libc::fprintf(fout, c"# %s\n".as_ptr(), obj.name.as_ptr()) };
        }
        if iobj_off(obj.flags) != 0 {
            unsafe { libc::fprintf(fout, c"#Turned off, no rendering.\n".as_ptr()) };
            continue;
        }
        unsafe {
            libc::fprintf(fout, c"AttributeBegin\n".as_ptr());

            libc::fprintf(
                fout,
                c"Color [ %g %g %g  ]\n".as_ptr(),
                obj.red as std::ffi::c_double,
                obj.green as std::ffi::c_double,
                obj.blue as std::ffi::c_double,
            );

            opac = (100 - obj.trans as i32) as f32;
            opac = (opac as f64 * 0.01) as f32;
            libc::fprintf(
                fout,
                c"Opacity %g %g %g\n".as_ptr(),
                opac as std::ffi::c_double,
                opac as std::ffi::c_double,
                opac as std::ffi::c_double,
            );

            libc::fprintf(
                fout,
                c"LightSource \"ambientlight\" 1 \"intensity\" %g\n".as_ptr(),
                (obj.ambient as f64 / 256.0) as f32 as std::ffi::c_double,
            );

            libc::fprintf(
                fout,
                c"Surface \"plastic\" \"Ks\" %g\n".as_ptr(),
                (obj.shininess as f32 * 0.007_812_5f32) as std::ffi::c_double,
            );
        }

        if iobj_close(obj.flags) != 0 {
            for m in 0..obj.mesh.len() {
                p_rib_mesh(fout, &obj.mesh[m], imod.zscale as f64);
            }
        }
        if iobj_scat(obj.flags) != 0 {
            p_rib_scat(fout, obj, imod.zscale as f64);
        }
        if iobj_open(obj.flags) != 0 {
            p_rib_tubes(fout, obj, imod.zscale as f64);
        }

        unsafe { libc::fprintf(fout, c"AttributeEnd\n".as_ptr()) };
    }
    unsafe { libc::fprintf(fout, c"WorldEnd\n".as_ptr()) };
    0
}

/// Original: `pRIB_mesh` (`imodel_to.c:640`).
///
/// `imodMeshPolyNormFactors` (`imesh.c:312`) has no translated module yet, so
/// its three index factors are computed in place.
pub fn p_rib_mesh(fout: *mut libc::FILE, mesh: &Imesh, zscale: f64) -> i32 {
    let mut cndat = Ipoint::default();
    let mut norm = [Ipoint::default(); 3];
    let mut vert = [Ipoint::default(); 3];
    let z = zscale as f32;

    if mesh.list.is_empty() {
        return -1;
    }
    let lsize = mesh.list.len();

    cndat.x = 0.0;
    cndat.y = 0.0;
    cndat.z = 1.0;
    let mut cnormal = cndat;

    let mut i = 0usize;
    while i < lsize {
        match mesh.list[i] {
            /* IMOD_MESH_BGNPOLY */
            -21 => {
                unsafe { libc::fprintf(fout, c"Polygon \"P\" [".as_ptr()) };
                loop {
                    i += 1;
                    if mesh.list[i] == IMOD_MESH_ENDPOLY {
                        break;
                    }
                    let v = mesh.vert[mesh.list[i] as usize];
                    unsafe {
                        libc::fprintf(
                            fout,
                            c" %g %g %g ".as_ptr(),
                            v.x as std::ffi::c_double,
                            v.y as std::ffi::c_double,
                            (v.z * z) as std::ffi::c_double,
                        )
                    };
                }
                unsafe { libc::fprintf(fout, c"]\n".as_ptr()) };
            }

            /* IMOD_MESH_BGNBIGPOLY: todo: draw concave poly */
            -24 => loop {
                i += 1;
                if mesh.list[i] == IMOD_MESH_ENDPOLY {
                    break;
                }
            },

            IMOD_MESH_BGNPOLYNORM | IMOD_MESH_BGNPOLYNORM2 => {
                /* imodMeshPolyNormFactors (`imesh.c:312`) */
                let (list_inc, vert_base, norm_add) = if mesh.list[i] == IMOD_MESH_BGNPOLYNORM {
                    (2usize, 1usize, 0i32)
                } else {
                    (1usize, 0usize, 1i32)
                };
                i += 1;
                while mesh.list[i] != IMOD_MESH_ENDPOLY {
                    for j in 0..3 {
                        norm[j] = mesh.vert[mesh.list[i + vert_base] as usize];
                        vert[j] = mesh.vert[(mesh.list[i] + norm_add) as usize];
                        i += list_inc;
                    }

                    unsafe {
                        libc::fprintf(fout, c"Polygon \"P\" [".as_ptr());
                        for v in vert {
                            libc::fprintf(
                                fout,
                                c" %g %g %g ".as_ptr(),
                                v.x as std::ffi::c_double,
                                v.y as std::ffi::c_double,
                                (v.z * z) as std::ffi::c_double,
                            );
                        }
                        libc::fprintf(fout, c"] \"N\" [".as_ptr());
                        for n in norm {
                            libc::fprintf(
                                fout,
                                c" %g %g %g ".as_ptr(),
                                n.x as std::ffi::c_double,
                                n.y as std::ffi::c_double,
                                (n.z * z) as std::ffi::c_double,
                            );
                        }
                        libc::fprintf(fout, c"]\n".as_ptr());
                    }
                }
            }

            /* IMOD_MESH_SWAP */
            -10 => {}
            /* IMOD_MESH_BGNTRI */
            -11 => {
                loop {
                    i += 1;
                    /* IMOD_MESH_ENDTRI */
                    if mesh.list[i] == -12 {
                        break;
                    }
                }
            }
            /* IMOD_MESH_NORMAL */
            -20 => {
                i += 1;
                cnormal = mesh.vert[mesh.list[i] as usize];
            }
            IMOD_MESH_END => {
                i = lsize;
            }

            _ => {}
        }
        let _ = cnormal;
        i += 1;
    }

    0
}

/// Original: `pRIB_scat` (`imodel_to.c:731`).
pub fn p_rib_scat(fout: *mut libc::FILE, obj: &Iobj, z: f64) -> i32 {
    let ssize = obj.pdrawsize as f32 * 0.5f32;

    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        if cont.pts.is_empty() {
            continue;
        }
        for pt in 0..cont.pts.len() {
            unsafe {
                libc::fprintf(fout, c"TransformBegin\n".as_ptr());
                libc::fprintf(
                    fout,
                    c"Translate %g %g %g\n".as_ptr(),
                    cont.pts[pt].x as std::ffi::c_double,
                    cont.pts[pt].y as std::ffi::c_double,
                    cont.pts[pt].z as std::ffi::c_double * z,
                );
                libc::fprintf(
                    fout,
                    c"Scale %g %g %g\n".as_ptr(),
                    ssize as std::ffi::c_double,
                    ssize as std::ffi::c_double,
                    ssize as std::ffi::c_double,
                );
                libc::fprintf(fout, c"Sphere 1 -1 1 360\n".as_ptr());
                libc::fprintf(fout, c"TransformEnd\n".as_ptr());
            }
        }
    }
    0
}

/// Original: `pRIB_tubes` (`imodel_to.c:754`).
pub fn p_rib_tubes(fout: *mut libc::FILE, obj: &Iobj, z: f64) -> i32 {
    for co in 0..obj.cont.len() {
        let cont = &obj.cont[co];
        let lpt = cont.pts.len() as i32 - 1;
        if lpt <= 0 {
            continue;
        }

        for pt in 0..lpt as usize {
            prib_tube(
                fout,
                &cont.pts[pt],
                &cont.pts[pt + 1],
                8,
                obj.linewidth as i32,
                z,
            );
        }
    }
    0
}

/// Original: `prib_tube` (`imodel_to.c:771`).
pub fn prib_tube(
    fout: *mut libc::FILE,
    p1: &Ipoint,
    p2: &Ipoint,
    slices: i32,
    linewidth: i32,
    z: f64,
) -> i32 {
    let mut norm = Ipoint::default();
    let mut offset = [Ipoint::default(); 2];
    let Some(mut mat) = imod_mat_new(2) else {
        return -1;
    };

    norm.x = 0.0;
    norm.y = linewidth as f32;
    offset[0] = norm;
    let astep = (360 / slices) as f64;

    for _sl in 0..slices {
        imod_mat_rot(&mut mat, astep, 0);
        let mut rotated = Ipoint::default();
        imod_mat_transform(&mat, &norm, &mut rotated);
        offset[1] = rotated;

        unsafe {
            libc::fprintf(fout, c"Polygon \"P\" ".as_ptr());
            libc::fprintf(
                fout,
                c"[ %g %g %g  %g %g %g  %g %g %g  %g %g %g ]\n".as_ptr(),
                (p2.x + offset[0].x) as std::ffi::c_double,
                (p2.y + offset[0].y) as std::ffi::c_double,
                p2.z as std::ffi::c_double * z,
                (p2.x + offset[1].x) as std::ffi::c_double,
                (p2.y + offset[1].y) as std::ffi::c_double,
                p2.z as std::ffi::c_double * z,
                (p1.x + offset[1].x) as std::ffi::c_double,
                (p1.y + offset[1].y) as std::ffi::c_double,
                p1.z as std::ffi::c_double * z,
                (p1.x + offset[0].x) as std::ffi::c_double,
                (p1.y + offset[0].y) as std::ffi::c_double,
                p1.z as std::ffi::c_double * z,
            );
        }
        offset[0] = offset[1];
    }
    imod_mat_delete(&mut mat);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{Icont, Imod, Iobj, Ipoint};

    #[test]
    fn imod_to_wmod_has_c_source_layout_and_numbers() {
        let model = Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![Ipoint {
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                    }],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        let mut text = Vec::new();
        imod_to_wmod(&model, &mut text, "out.wimp").unwrap();
        assert_eq!(
            String::from_utf8(text).unwrap(),
            concat!(
                " Model file name........................out.wimp\n",
                " max # of object.......................    2\n",
                " # of node.............................    2\n",
                " # of object...........................    1\n",
                "  Object sequence : \n",
                "  Object #:           1\n",
                " # of point:           1\n",
                " Display switch:1  247\n",
                "     #    X       Y       Z      Mark    Label \n",
                "      1    1.00    2.00    3.00   0\n\n  END\n"
            )
        );
    }
}
