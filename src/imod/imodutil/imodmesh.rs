//! Translation of `IMOD/imodutil/imodmesh.c` -- add mesh data to a model.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{
    ImodFile, imod_backup_file, imod_copyright, imod_prog_name, imod_version, set_or_clear_flags,
};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libimod::imesh::{
    DEFAULT_FLOAT, DEFAULT_VALUE, IMESH_CAP_ALL, IMESH_CAP_END, IMESH_CAP_OFF,
    IMESH_FLAG_RES_SHIFT, IMESH_MK_CAP_DOME, IMESH_MK_CAP_TUBE, IMESH_MK_FAST, IMESH_MK_NORM,
    IMESH_MK_SKIP, IMESH_MK_STRAY, IMESH_MK_SURF, IMESH_MK_TIME, IMESH_MK_TUBE,
    IMOD_MESH_BGNBIGPOLY, IMOD_MESH_BGNPOLY, IMOD_MESH_BGNPOLYNORM, IMOD_MESH_BGNPOLYNORM2,
    IMOD_MESH_BGNTRI, IMOD_MESH_END, IMOD_MESH_ENDPOLY, IMOD_MESH_ENDTRI, IMOD_MESH_NORMAL,
    IMOD_MESH_SWAP, imesh_copy_skip_list, imesh_params_new, imesh_resol,
    imod_mesh_poly_norm_factors, imodel_mesh_add,
};
use crate::imod::libimod::imodel::{Imesh, Imod, Ipoint, imodel_maxpt};
use crate::imod::libimod::imodel_files::{imod_read, imod_write};
use crate::imod::libimod::iobj::{
    IMOD_OBJFLAG_FILL, IMOD_OBJFLAG_MESH, IMOD_OBJFLAG_NOLINE, iobj_scat,
};
use crate::imod::libimod::ipoint::imod_point_normalize;
use crate::imod::libmesh::objprep::{analyze_prep_skin_obj, imod_meshes_delete_res};
use crate::imod::libmesh::remesh::{imesh_remesh_normal, imesh_set_new_poly_norm};

/// Original: `DEFAULT_TOL` (`imodmesh.c:30`).
const DEFAULT_TOL: f32 = 0.25;
/// Original: `DEFAULT_FLAT` (`imodmesh.c:31`).
const DEFAULT_FLAT: f32 = 1.5;

/// Original static `imodmesh_usage` (`imodmesh.c:33`).
pub fn imodmesh_usage(prog: &str, retcode: i32) -> i32 {
    let mut err = ImodFile::Stderr;
    let _ = err.write_all(format!("{prog} usage: {prog} [options] [input files...]\n").as_bytes());
    let _ = err.write_all(b"options:\n");
    let _ = err.write_all(b"\t-u  \tUse stored meshing parameters.\n");
    let _ = err.write_all(b"\t-c  \tCap ends of object.\n");
    let _ = err.write_all(b"\t-C  \tCap all unconnected ends of object.\n");
    let _ = err.write_all(b"\t-D list\tDo not cap ends unconnected to Z values in the list.\n");
    let _ = err.write_all(b"\t-p #\tOnly connect contours that overlap by the given percentage.\n");
    let _ = err.write_all(b"\t-s  \tConnect contours across sections with no contours.\n");
    let _ = err.write_all(b"\t-P #\tDo multiple passes to connect contours farther apart in Z.\n");
    let _ = err.write_all(b"\t-S  \tUse surface numbers for connections.\n");
    let _ = err.write_all(b"\t-F #\tCriterion Z difference for analyzing for tilted contours.\n");
    let _ = err.write_all(b"\t-I  \tIgnore time values and connect across times.\n");
    let _ = err.write_all(b"\t-f  \tForce more connections to non-overlapping contours.\n");
    let _ = err.write_all(b"\t-t list\tRender open contour objects in list as tubes.\n");
    let _ = err.write_all(b"\t-d #\tSet diameter for tubes (default is 3D line width).\n");
    let _ = err.write_all(b"\t-E  \tCap ends of tubes.\n");
    let _ = err.write_all(b"\t-H  \tCap ends of tubes with hemispheres.\n");
    let _ = err.write_all(b"\t-T  \tDo time consuming calculations, may help reduce artifacts.\n");

    let _ = err.write_all(b"\t-o list\tDo operations only on objects in list (ranges allowed).\n");
    let _ = err.write_all(
        format!(
            "\t-R #\tTolerance (maximum error) for point reduction (default {:.2}).\n",
            DEFAULT_TOL
        )
        .as_bytes(),
    );

    let _ = err.write_all(b"\t-i #\tOnly mesh sections at the given z increment.\n");
    let _ = err
        .write_all(b"\t-z #,#,#  Only mesh sections within the given z range at a z increment.\n");
    let _ = err.write_all(b"\t-l \tMark new meshes as low-resolution and keep high res meshes.\n");
    let _ = err.write_all(b"\t-x #,#\tClip mesh outside given lower and upper limits in X.\n");
    let _ = err.write_all(b"\t-y #,#\tClip mesh outside given lower and upper limits in Y.\n");
    let _ = err.write_all(b"\t-no* \tOverride a stored parameter; * is one of cCDsSIftEHTxyz.\n");

    let _ = err.write_all(b"\t-a \tAppend mesh data to object, replacing mesh in same z range.\n");
    let _ = err.write_all(b"\t-e  \tErase meshes, overrides other options.\n");
    let _ = err.write_all(b"\t-N \tRecompute normals for existing meshes.\n");
    let _ = err.write_all(b"\t-n \tRescale normals using the -Z value for existing meshes.\n");
    let _ = err.write_all(b"\t-Z #\tNormal scaling z multiplier.\n");
    let _ = err.write_all(b"\t-B  \tMake mesh backward-compatible to IMOD before 3.6.14.\n");
    if retcode != 0 {
        std::process::exit(retcode);
    }
    retcode
}

/// Original: `DELETE_INDEX` (`imodmesh.c:92`).
const DELETE_INDEX: i32 = -123456;
/// Original: `LOWRES_INCZ` (`imodmesh.c:93`).
const LOWRES_INCZ: i32 = 4;
/// Original: `LOWRES_TOL` (`imodmesh.c:94`).
const LOWRES_TOL: f32 = 2.;

/// Original: `main` (`imodmesh.c:96`).
pub fn imodmesh() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;
    let mut spnt = Ipoint::default();
    let mut max = Ipoint::default();
    let mut erase_mesh = false;
    let mut renorm = false;
    let mut remesh_norm = false;
    let mut cap = IMESH_CAP_OFF;
    let mut times = true;
    let mut use_old_param = false;
    let mut notimes = false;
    let mut nocap = false;
    let mut no_skip_list = false;
    let mut noxmin = false;
    let mut noymin = false;
    let mut nozmin = false;
    let mut notube = false;
    let mut dowarn1 = true;
    let mut dowarn2 = true;
    let mut dowarn3 = true;
    let mut set_flags: u32 = IMESH_MK_NORM | IMESH_MK_TIME;
    let mut clear_flags: u32 = 0;
    let mut overlap: f32 = -1.;
    let mut tube_diameter: f32 = -99.;
    let mut flat_crit: f32 = -1.;
    let mut tol: f32 = -1.;
    let mut resol: i32 = 0;
    let mut passes: i32 = -1;

    /* Added 2.00 mesh only given sections. */
    let mut minz: i32 = DEFAULT_VALUE;
    let mut maxz: i32 = DEFAULT_VALUE;
    let mut incz: i32 = -1;
    let mut zscale: f32 = 1.0;
    let mut append = false;
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

    let mut new_poly_norm: i32 = 1;
    let mut nlist: i32 = 0;
    let mut list: Vec<i32> = Vec::new();
    let mut cap_skip_nz: i32 = 0;
    let mut cap_skip_zlist: Option<Vec<i32>> = None;
    let mut ntube_list: i32 = 0;
    let mut tube_list: Vec<i32> = Vec::new();
    let progname = imod_prog_name(&argv[0]);
    let oldvar = "IMODMESH_OLDMESH";

    if argc < 1 {
        imod_version(Some(&progname));
        imod_copyright();
        imodmesh_usage(&progname, -1);
    }

    let mut iarg: usize = 1;
    while (iarg as i32) < argc {
        let arg = argv[iarg].as_bytes();
        if !arg.is_empty() && arg[0] == b'-' {
            if !(arg.len() > 2 && arg[1] == b'n' && arg[2] == b'o') {
                match *arg.get(1).unwrap_or(&0) {
                    b'd' => {
                        iarg += 1;
                        tube_diameter = atof(&argv[iarg]) as f32;
                    }
                    b'o' => {
                        /* select object list */
                        iarg += 1;
                        match parselist(&argv[iarg]) {
                            Ok(parsed) => {
                                nlist = parsed.len() as i32;
                                list = parsed;
                            }
                            Err(_) => {
                                let _ = ImodFile::Stderr.write_all(
                                    format!("{progname}: Error parsing object list\n").as_bytes(),
                                );
                                std::process::exit(3);
                            }
                        }
                    }
                    /* 9/7/06: Removed -r option */
                    b'R' => {
                        iarg += 1;
                        tol = atof(&argv[iarg]) as f32;
                        if tol < 0. {
                            tol = 0.;
                        }
                    }
                    b'F' => {
                        iarg += 1;
                        flat_crit = atof(&argv[iarg]) as f32;
                    }
                    b'c' => cap = IMESH_CAP_END,
                    b'C' => cap = IMESH_CAP_ALL,
                    b'D' => {
                        /* Do not cap at Z in list. */
                        iarg += 1;
                        match parselist(&argv[iarg]) {
                            Ok(parsed) => {
                                cap_skip_nz = parsed.len() as i32;
                                cap_skip_zlist = Some(parsed);
                            }
                            Err(_) => {
                                let _ = ImodFile::Stderr.write_all(
                                    format!("{progname}: Error parsing Z list\n").as_bytes(),
                                );
                                std::process::exit(3);
                            }
                        }
                    }
                    b'h' => {
                        imodmesh_usage(&progname, -1);
                    }
                    b'e' => erase_mesh = true, /* erase meshes */
                    b'f' => set_flags |= IMESH_MK_STRAY,
                    b'E' => set_flags |= IMESH_MK_CAP_TUBE,
                    b'H' => set_flags |= IMESH_MK_CAP_DOME,
                    b'T' => set_flags |= IMESH_MK_FAST,
                    b's' => set_flags |= IMESH_MK_SKIP,
                    b'S' => set_flags |= IMESH_MK_SURF,
                    b'I' => times = false,

                    b'n' => renorm = true,
                    b'N' => remesh_norm = true,

                    b't' => {
                        iarg += 1;
                        match parselist(&argv[iarg]) {
                            Ok(parsed) => {
                                ntube_list = parsed.len() as i32;
                                tube_list = parsed;
                            }
                            Err(_) => {
                                let _ = ImodFile::Stderr.write_all(
                                    format!(
                                        "{progname}: Error parsing list of objects to mesh as tubes\n"
                                    )
                                    .as_bytes(),
                                );
                                std::process::exit(3);
                            }
                        }
                    }

                    b'z' => {
                        iarg += 1;
                        sscanf_d_any_d_any_d(&argv[iarg], &mut minz, &mut maxz, &mut incz);
                    }
                    b'i' => {
                        iarg += 1;
                        incz = atoi(&argv[iarg]);
                        if incz < 1 {
                            incz = 1;
                        }
                    }
                    b'x' => {
                        iarg += 1;
                        sscanf_f_any_f(&argv[iarg], &mut tri_min.x, &mut tri_max.x);
                    }
                    b'y' => {
                        iarg += 1;
                        sscanf_f_any_f(&argv[iarg], &mut tri_min.y, &mut tri_max.y);
                    }
                    b'Z' => {
                        iarg += 1;
                        zscale = atof(&argv[iarg]) as f32;
                    }
                    b'a' => append = true,
                    b'p' => {
                        iarg += 1;
                        overlap = (atof(&argv[iarg]) / 100.) as f32;
                    }
                    b'l' => resol = 1,
                    b'P' => {
                        iarg += 1;
                        passes = atoi(&argv[iarg]);
                        if passes < 1 {
                            passes = 1;
                        }
                    }
                    b'B' => new_poly_norm = 0,
                    b'u' => use_old_param = true,

                    b'?' => {
                        imodmesh_usage(&progname, -1);
                    }
                    _ => {
                        let _ = ImodFile::Stderr.write_all(
                            format!("{progname}: unknown option {}\n", argv[iarg]).as_bytes(),
                        );
                        imodmesh_usage(&progname, -1);
                    }
                }
            } else {
                match *arg.get(3).unwrap_or(&0) {
                    b'c' | b'C' => nocap = true,
                    b'D' => no_skip_list = true,
                    b'f' => clear_flags |= IMESH_MK_STRAY,
                    b'E' => clear_flags |= IMESH_MK_CAP_TUBE,
                    b'H' => clear_flags |= IMESH_MK_CAP_DOME,
                    b'T' => clear_flags |= IMESH_MK_FAST,
                    b's' => clear_flags |= IMESH_MK_SKIP,
                    b'S' => clear_flags |= IMESH_MK_SURF,
                    b'I' => notimes = true,
                    b'x' => noxmin = true,
                    b'y' => noymin = true,
                    b'z' => nozmin = true,
                    b't' => notube = true,
                    _ => {
                        let _ = ImodFile::Stderr.write_all(
                            format!("{progname}: unknown option {}\n", argv[iarg]).as_bytes(),
                        );
                        imodmesh_usage(&progname, -1);
                    }
                }
            }
        } else {
            break;
        }
        iarg += 1;
    }

    if iarg as i32 >= argc {
        imodmesh_usage(&progname, -1);
    }

    if (set_flags & clear_flags) != 0
        || (!times && notimes)
        || (ntube_list != 0 && notube)
        || (nocap && cap != IMESH_CAP_OFF)
        || (no_skip_list && cap_skip_nz != 0)
        || (nozmin && minz != DEFAULT_VALUE)
        || (noxmin && tri_min.x > -DEFAULT_FLOAT)
        || (noymin && tri_min.y > -DEFAULT_FLOAT)
    {
        let _ = ImodFile::Stderr.write_all(
            format!("{progname}: You cannot enter an option and its 'no' variation\n").as_bytes(),
        );
        std::process::exit(1);
    }

    if times {
        set_flags |= IMESH_MK_TIME;
        clear_flags &= !IMESH_MK_TIME;
    } else {
        clear_flags |= IMESH_MK_TIME;
        set_flags &= !IMESH_MK_TIME;
    }

    if new_poly_norm != 0 && std::env::var_os(oldvar).is_some() {
        new_poly_norm = 0;
        let _ = ImodFile::Stdout.write_all(
            format!("Making backward-compatible meshes because {oldvar} is set in environment\n")
                .as_bytes(),
        );
    }
    imesh_set_new_poly_norm(new_poly_norm);

    /* Skin multiple models in serial. */
    while (iarg as i32) < argc {
        let _ = ImodFile::Stdout.write_all(format!("Model: {}\n", argv[iarg]).as_bytes());

        let Ok(mut imod) = imod_read(&argv[iarg]) else {
            let _ = ImodFile::Stderr
                .write_all(format!("{progname}: Error reading model {}\n", argv[iarg]).as_bytes());
            std::process::exit(3);
        };

        spnt.x = imod.xscale;
        spnt.y = imod.yscale;
        spnt.z = imod.zscale * zscale;
        imodel_maxpt(&imod, &mut max);

        if erase_mesh {
            let _ = ImodFile::Stdout.write_all(b"Erasing mesh from objects.\n");
        }

        for ob in 0..imod.obj.len() {
            if imod.obj[ob].cont.is_empty() && !erase_mesh {
                continue;
            }
            if obj_on_list(ob as i32, &list, nlist) != 0 {
                if erase_mesh {
                    if !imod.obj[ob].mesh.is_empty() {
                        let mut meshsize = imod.obj[ob].mesh.len() as i32;
                        imod_meshes_delete_z_range(
                            &mut imod.obj[ob].mesh,
                            &mut meshsize,
                            minz,
                            maxz,
                            resol,
                        );

                        /* Turn off standard mesh view flags if they are all on and the
                        regular resolution is being erased.  Have to modify views too */
                        let flags = IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL;
                        if resol == 0 && (imod.obj[ob].flags & flags) == flags {
                            let mut objflags = imod.obj[ob].flags;
                            set_or_clear_flags(&mut objflags, flags, 0);
                            imod.obj[ob].flags = objflags;
                        }
                        for iv in 1..imod.view.len() {
                            if ob < imod.view[iv].objview.len() {
                                let mut obvflags = imod.view[iv].objview[ob].flags;
                                if resol == 0 && (obvflags & flags) == flags {
                                    set_or_clear_flags(&mut obvflags, flags, 0);
                                    imod.view[iv].objview[ob].flags = obvflags;
                                }
                            }
                        }
                    }
                } else if renorm {
                    spnt.x = 1.0f32;
                    spnt.y = 1.0f32;
                    spnt.z = zscale;
                    let mut meshsize = imod.obj[ob].mesh.len() as i32;
                    imod_meshes_rescale_normal(&mut imod.obj[ob].mesh, &mut meshsize, &spnt);
                } else if remesh_norm {
                    if !imod.obj[ob].mesh.is_empty() {
                        let mut meshsize = imod.obj[ob].mesh.len() as i32;
                        let mut meshes = std::mem::take(&mut imod.obj[ob].mesh);
                        imod.obj[ob].mesh =
                            imesh_remesh_normal(&mut meshes, &mut meshsize, Some(&spnt), resol)
                                .unwrap_or_default();
                    }
                } else if iobj_scat(imod.obj[ob].flags) == 0 {
                    let use_param = i32::from(use_old_param && imod.obj[ob].mesh_param.is_some());
                    if imod.obj[ob].mesh_param.is_none() {
                        match imesh_params_new() {
                            Some(params) => imod.obj[ob].mesh_param = Some(params),
                            None => {
                                let _ = ImodFile::Stderr.write_all(
                                    format!("{progname}: Error creating new parameter structure\n")
                                        .as_bytes(),
                                );
                                std::process::exit(3);
                            }
                        }
                    }
                    let param = imod.obj[ob].mesh_param.as_mut().unwrap();

                    if use_param != 0 {
                        param.flags = (param.flags & !clear_flags) | set_flags;
                    } else {
                        param.flags = set_flags;
                    }

                    if use_param == 0 || ntube_list > 0 || notube {
                        if ntube_list > 0 && obj_on_list(ob as i32, &tube_list, ntube_list) != 0 {
                            param.flags |= IMESH_MK_TUBE;
                        } else {
                            param.flags &= !IMESH_MK_TUBE;
                        }
                    }

                    if use_param == 0 || minz != DEFAULT_VALUE || nozmin {
                        param.minz = minz;
                        param.maxz = maxz;
                    }

                    if use_param == 0 || incz >= 0 {
                        if resol != 0 {
                            param.incz_low_res = incz;
                            if param.incz_low_res < 0 {
                                param.incz_low_res = LOWRES_INCZ;
                                if dowarn1 {
                                    let _ = ImodFile::Stdout.write_all(
                                        format!(
                                            "Setting Z increment to {} for low resolution mesh\n",
                                            param.incz_low_res
                                        )
                                        .as_bytes(),
                                    );
                                }
                                dowarn1 = false;
                            }
                        } else {
                            param.incz_high_res = 1.max(incz);
                        }
                    }

                    if use_param == 0 || tol >= 0. {
                        if resol != 0 {
                            param.tol_low_res = tol;
                            if param.tol_low_res < 0. {
                                param.tol_low_res = LOWRES_TOL;
                                if dowarn2 {
                                    let _ = ImodFile::Stdout.write_all(
                                        format!(
                                            "Setting tolerance to {:.2} for low resolution mesh\n",
                                            param.tol_low_res
                                        )
                                        .as_bytes(),
                                    );
                                }
                                dowarn2 = false;
                            }
                        } else {
                            param.tol_high_res = tol;
                            if param.tol_high_res < 0. {
                                param.tol_high_res = DEFAULT_TOL;
                                if dowarn3 {
                                    let _ = ImodFile::Stdout.write_all(
                                        format!(
                                            "Setting tolerance to {:.2} for point reduction\n",
                                            param.tol_high_res
                                        )
                                        .as_bytes(),
                                    );
                                }
                                dowarn3 = false;
                            }
                        }
                    }

                    if (use_param == 0 || tube_diameter >= -10.)
                        && (param.flags & IMESH_MK_TUBE) != 0
                    {
                        param.tube_diameter = if -2.0f32 > tube_diameter {
                            -2.0f32
                        } else {
                            tube_diameter
                        };
                    }

                    if use_param == 0 || flat_crit >= 0. {
                        param.flat_crit = flat_crit;
                    }
                    if param.flat_crit < 0. {
                        param.flat_crit = DEFAULT_FLAT;
                    }

                    if use_param == 0 || overlap >= 0. {
                        param.overlap = if 0.0f32 > overlap { 0.0f32 } else { overlap };
                    }

                    if use_param == 0 || passes >= 0 {
                        param.passes = 1.max(passes);
                    }

                    if use_param == 0 || cap != IMESH_CAP_OFF || nocap {
                        param.cap = cap;
                    }

                    if use_param == 0 || tri_min.x > -DEFAULT_FLOAT || noxmin {
                        param.xmin = tri_min.x;
                        param.xmax = tri_max.x;
                    }
                    if use_param == 0 || tri_min.y > -DEFAULT_FLOAT || noymin {
                        param.ymin = tri_min.y;
                        param.ymax = tri_max.y;
                    }

                    if use_param == 0 || cap_skip_nz != 0 || no_skip_list {
                        let mut nto = 0;
                        let mut lto: Option<Vec<i32>> = param.cap_skip_zlist.take();
                        let mut old_nto = param.cap_skip_nz;
                        if imesh_copy_skip_list(
                            cap_skip_zlist.as_deref(),
                            cap_skip_nz,
                            &mut lto,
                            &mut old_nto,
                        ) != 0
                        {
                            std::process::exit(3);
                        }
                        nto = old_nto;
                        param.cap_skip_zlist = lto;
                        param.cap_skip_nz = nto;

                        /* If there is a list of capping exclusion Z values, check if 0
                        and model max are on list, and extend range by 1 */
                        if let Some(zlist) = param.cap_skip_zlist.as_mut() {
                            for ilist in 0..param.cap_skip_nz as usize {
                                if zlist[ilist] == 0 {
                                    zlist[param.cap_skip_nz as usize] = -1;
                                    param.cap_skip_nz += 1;
                                    break;
                                }
                            }
                            for ilist in 0..param.cap_skip_nz as usize {
                                if zlist[ilist] as f64 == (max.z as f64 + 0.5).floor() {
                                    zlist[param.cap_skip_nz as usize] =
                                        (max.z as f64 + 1.5).floor() as i32;
                                    param.cap_skip_nz += 1;
                                    break;
                                }
                            }
                        }
                    }

                    let mut tmshsize: i32 = 0;
                    let mut tmsh: Vec<Imesh> = Vec::new();
                    if !imod.obj[ob].mesh.is_empty() {
                        /* DNM: if appending, delete defined range and save mesh to add
                        to later */
                        let mut meshsize = imod.obj[ob].mesh.len() as i32;
                        if append {
                            imod_meshes_delete_z_range(
                                &mut imod.obj[ob].mesh,
                                &mut meshsize,
                                minz,
                                maxz,
                                resol,
                            );
                        } else {
                            imod_meshes_delete_res(&mut imod.obj[ob].mesh, &mut meshsize, resol);
                        }
                        tmsh = std::mem::take(&mut imod.obj[ob].mesh);
                        tmshsize = tmsh.len() as i32;
                    }
                    let _ = ImodFile::Stdout
                        .write_all(format!("Meshing object  {}\n", ob + 1).as_bytes());
                    let _ = ImodFile::Stdout.flush();
                    if analyze_prep_skin_obj(&mut imod.obj[ob], resol, &spnt, None) != 0 {
                        let _ = ImodFile::Stderr.write_all(
                            format!("{progname}: meshing error in model {}\n", argv[iarg])
                                .as_bytes(),
                        );
                        std::process::exit(3);
                    }

                    /* set resolution flag now */
                    for m in 0..imod.obj[ob].mesh.len() {
                        imod.obj[ob].mesh[m].flag |= (resol as u32) << IMESH_FLAG_RES_SHIFT;
                    }

                    if tmshsize != 0 {
                        /* If appending, add new mesh to saved mesh then put it back
                        into this object's mesh */
                        for m in 0..imod.obj[ob].mesh.len() {
                            let mesh = imod.obj[ob].mesh[m].clone();
                            imodel_mesh_add(Some(&mesh), &mut tmsh);
                        }
                        let mut tmshsize = tmsh.len() as i32;
                        if append {
                            imod.obj[ob].mesh =
                                imesh_remesh_normal(&mut tmsh, &mut tmshsize, Some(&spnt), resol)
                                    .unwrap_or_default();
                        } else {
                            imod.obj[ob].mesh = tmsh;
                        }
                    }

                    /* Turn on standard mesh view flags if none of them are on,
                    otherwise turn on mesh flag */
                    let flags = IMOD_OBJFLAG_MESH | IMOD_OBJFLAG_NOLINE | IMOD_OBJFLAG_FILL;
                    let mut objflags = imod.obj[ob].flags;
                    if objflags & flags != 0 {
                        set_or_clear_flags(&mut objflags, IMOD_OBJFLAG_MESH, 1);
                    } else {
                        set_or_clear_flags(&mut objflags, flags, 1);
                    }
                    imod.obj[ob].flags = objflags;
                    for iv in 1..imod.view.len() {
                        if ob < imod.view[iv].objview.len() {
                            let mut obvflags = imod.view[iv].objview[ob].flags;
                            if obvflags & flags != 0 {
                                set_or_clear_flags(&mut obvflags, IMOD_OBJFLAG_MESH, 1);
                            } else {
                                set_or_clear_flags(&mut obvflags, flags, 1);
                            }
                            imod.view[iv].objview[ob].flags = obvflags;
                        }
                    }
                } else {
                    let _ = ImodFile::Stdout
                        .write_all(format!("Skipping object {}\n", ob + 1).as_bytes());
                }
            }
        }

        /* Save backup of Model to Model~ */
        if imod_backup_file(&argv[iarg]) != 0 {
            let _ = ImodFile::Stderr
                .write_all(format!("{progname}: Error, couldn't create backup").as_bytes());
            std::process::exit(3);
        }

        let Some(mut fout) = ImodFile::open(&argv[iarg], "wb") else {
            let _ = ImodFile::Stderr
                .write_all(format!("{progname}: Error, couldn't open output.").as_bytes());
            std::process::exit(3);
        };

        let _ = imod_write(&imod, &mut fout);
        drop(fout);
        drop(imod);
        iarg += 1;
    }
    std::process::exit(0);
}

/// Original static `imodMeshesDeleteZRange` (`imodmesh.c:571`).
pub fn imod_meshes_delete_z_range(
    meshp: &mut Vec<Imesh>,
    meshsize: &mut i32,
    minz: i32,
    maxz: i32,
    resol: i32,
) {
    if minz == DEFAULT_VALUE && maxz == DEFAULT_VALUE {
        imod_meshes_delete_res(meshp, meshsize, resol);
        return;
    }

    for m in 0..*meshsize as usize {
        if imesh_resol(meshp[m].flag) == resol {
            imod_mesh_delete_z_range(&mut meshp[m], minz, maxz);
        }
    }
}

/// Original static `deletez` (`imodmesh.c:587`).
pub fn deletez(point: Ipoint, minz: i32, maxz: i32) -> i32 {
    let mut rmpoint = 0;
    let mut rmlow = 0;
    let mut rmup = 0;
    let z = (point.z + 0.5f32) as i32;

    if minz != DEFAULT_VALUE && z >= minz {
        rmlow = 1;
    }

    if maxz != DEFAULT_VALUE && z <= maxz {
        rmup = 1;
    }
    if rmlow != 0 && rmup != 0 {
        rmpoint = 1;
    }

    rmpoint
}

/// Original static `imodMeshDeleteZRange` (`imodmesh.c:610`).
///
/// Deletes geometries in the range of `minz` and `maxz`.  The vertex data is
/// not changed, only the index array is changed.
pub fn imod_mesh_delete_z_range(mesh: &mut Imesh, minz: i32, maxz: i32) {
    let mut pt = [Ipoint::default(); 6];
    let mut newsize = 0usize;

    let mut i: i32 = 0;
    while i < mesh.list.len() as i32 {
        match mesh.list[i as usize] {
            IMOD_MESH_NORMAL => {
                i += 1;
            }

            IMOD_MESH_BGNPOLYNORM => {
                i += 1;
                while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                    if i + 7 > mesh.list.len() as i32 {
                        break;
                    }
                    pt[0] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[1] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[2] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[3] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[4] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[5] = mesh.vert[mesh.list[i as usize] as usize];

                    if deletez(pt[1], minz, maxz) != 0
                        && deletez(pt[3], minz, maxz) != 0
                        && deletez(pt[5], minz, maxz) != 0
                    {
                        i -= 5;
                        for _ in 0..5 {
                            mesh.list[i as usize] = DELETE_INDEX;
                            i += 1;
                        }
                        mesh.list[i as usize] = DELETE_INDEX;
                    }
                    i += 1;
                }
            }

            IMOD_MESH_BGNPOLYNORM2 => {
                i += 1;
                while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                    if i + 4 > mesh.list.len() as i32 {
                        break;
                    }
                    pt[1] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[3] = mesh.vert[mesh.list[i as usize] as usize];
                    i += 1;
                    pt[5] = mesh.vert[mesh.list[i as usize] as usize];

                    if deletez(pt[1], minz, maxz) != 0
                        && deletez(pt[3], minz, maxz) != 0
                        && deletez(pt[5], minz, maxz) != 0
                    {
                        i -= 2;
                        mesh.list[i as usize] = DELETE_INDEX;
                        i += 1;
                        mesh.list[i as usize] = DELETE_INDEX;
                        i += 1;
                        mesh.list[i as usize] = DELETE_INDEX;
                    }
                    i += 1;
                }
            }

            IMOD_MESH_END | IMOD_MESH_SWAP | IMOD_MESH_ENDTRI | IMOD_MESH_BGNTRI => {}

            _ => {}
        }
        i += 1;
    }

    for i in 0..mesh.list.len() {
        if mesh.list[i] != DELETE_INDEX {
            mesh.list[newsize] = mesh.list[i];
            newsize += 1;
        }
    }
    mesh.list.truncate(newsize);
}

/// Original static `imodMeshesRescaleNormal` (`imodmesh.c:706`).
pub fn imod_meshes_rescale_normal(in_mesh: &mut [Imesh], meshsize: &mut i32, spnt: &Ipoint) {
    let mut list_inc: i32 = 0;
    let mut vert_base: i32 = 0;
    let mut norm_add: i32 = 0;

    if in_mesh.is_empty() || *meshsize < 1 {
        return;
    }

    for m in 0..*meshsize as usize {
        let mesh = &mut in_mesh[m];
        let lsize = mesh.list.len() as i32;

        let mut i: i32 = 0;
        while i < lsize {
            match mesh.list[i as usize] {
                IMOD_MESH_BGNPOLY | IMOD_MESH_BGNBIGPOLY => {
                    i += 1;
                    while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                        i += 1;
                    }
                }

                IMOD_MESH_NORMAL => {
                    i += 1;
                    if (mesh.list[i as usize] < mesh.vert.len() as i32)
                        && (mesh.list[i as usize] > -1)
                    {
                        let ind = mesh.list[i as usize] as usize;
                        mesh.vert[ind].x *= spnt.x;
                        mesh.vert[ind].y *= spnt.y;
                        mesh.vert[ind].z *= spnt.z;
                        let mut v = mesh.vert[ind];
                        imod_point_normalize(&mut v);
                        mesh.vert[ind] = v;
                    }
                }

                IMOD_MESH_BGNPOLYNORM | IMOD_MESH_BGNPOLYNORM2 => {
                    imod_mesh_poly_norm_factors(
                        mesh.list[i as usize],
                        &mut list_inc,
                        &mut vert_base,
                        &mut norm_add,
                    );
                    i += 1;
                    while mesh.list[i as usize] != IMOD_MESH_ENDPOLY {
                        for _j in 0..3 {
                            let ind = (mesh.list[i as usize] + norm_add) as usize;
                            mesh.vert[ind].x *= spnt.x;
                            mesh.vert[ind].y *= spnt.y;
                            mesh.vert[ind].z *= spnt.z;
                            let mut v = mesh.vert[ind];
                            imod_point_normalize(&mut v);
                            mesh.vert[ind] = v;
                            i += list_inc;
                        }
                    }
                }

                IMOD_MESH_BGNTRI | IMOD_MESH_ENDTRI | IMOD_MESH_SWAP => {}

                _ => {}
            }
            i += 1;
        }
    }
}

/// Original static `ObjOnList` (`imodmesh.c:774`).
pub fn obj_on_list(ob: i32, list: &[i32], nlist: i32) -> i32 {
    if nlist == 0 {
        return 1;
    }
    for i in 0..nlist as usize {
        if ob + 1 == list[i] {
            return 1;
        }
    }
    0
}

/// C `atof`: parses a leading floating-point number, 0.0 on failure.
fn atof(value: &str) -> f64 {
    let trimmed = value.trim_start();
    let bytes = trimmed.as_bytes();
    let mut end = 0;
    if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
        end += 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end < bytes.len() && bytes[end] == b'.' {
        end += 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }
    let mantissa_end = end;
    if end < bytes.len() && (bytes[end] == b'e' || bytes[end] == b'E') {
        let mut exp = end + 1;
        if exp < bytes.len() && (bytes[exp] == b'+' || bytes[exp] == b'-') {
            exp += 1;
        }
        let digit_start = exp;
        while exp < bytes.len() && bytes[exp].is_ascii_digit() {
            exp += 1;
        }
        if exp > digit_start {
            end = exp;
        }
    }
    let _ = mantissa_end;
    trimmed[..end].parse::<f64>().unwrap_or(0.0)
}

/// C `atoi`: parses a leading integer, 0 on failure.
fn atoi(value: &str) -> i32 {
    let trimmed = value.trim_start();
    let bytes = trimmed.as_bytes();
    let mut end = 0;
    if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
        end += 1;
    }
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    trimmed[..end].parse::<i32>().unwrap_or(0)
}

/// `sscanf(s, "%d%*c%d%*c%d", a, b, c)` (`imodmesh.c:236`): each conversion
/// that does not match leaves its variable unchanged, exactly as the C does.
fn sscanf_d_any_d_any_d(s: &str, a: &mut i32, b: &mut i32, c: &mut i32) {
    let bytes = s.as_bytes();
    let mut pos = 0usize;
    for (n, out) in [a, b, c].into_iter().enumerate() {
        if n > 0 {
            /* %*c consumes exactly one character */
            if pos >= bytes.len() {
                return;
            }
            pos += 1;
        }
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
            pos += 1;
        }
        let digits = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos == digits {
            return;
        }
        match s[start..pos].parse::<i32>() {
            Ok(value) => *out = value,
            Err(_) => return,
        }
    }
}

/// `sscanf(s, "%f%*c%f", a, b)` (`imodmesh.c:245`).
fn sscanf_f_any_f(s: &str, a: &mut f32, b: &mut f32) {
    let bytes = s.as_bytes();
    let mut pos = 0usize;
    for (n, out) in [a, b].into_iter().enumerate() {
        if n > 0 {
            if pos >= bytes.len() {
                return;
            }
            pos += 1;
        }
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let start = pos;
        if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
            pos += 1;
        }
        let digits = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if pos < bytes.len() && bytes[pos] == b'.' {
            pos += 1;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
        }
        if pos == digits {
            return;
        }
        if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
            let mut exp = pos + 1;
            if exp < bytes.len() && (bytes[exp] == b'+' || bytes[exp] == b'-') {
                exp += 1;
            }
            let digit_start = exp;
            while exp < bytes.len() && bytes[exp].is_ascii_digit() {
                exp += 1;
            }
            if exp > digit_start {
                pos = exp;
            }
        }
        match s[start..pos].parse::<f32>() {
            Ok(value) => *out = value,
            Err(_) => return,
        }
    }
}
