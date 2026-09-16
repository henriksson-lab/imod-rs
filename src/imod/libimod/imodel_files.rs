//! Binary model-file routines from `IMOD/libimod/imodel_files.c`.
//!
//! IMOD binary chunks are big-endian irrespective of the host byte order.

use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::imodel::{
    IMOD_CLIPSIZE, IMOD_ERROR_CORRUPT, IMOD_ERROR_FORMAT, IMOD_ERROR_MEMORY, IMOD_ERROR_READ,
    IMOD_ERROR_WRITE, IMOD_OBJFLAG_OFF, IMOD_OBJFLAG_OPEN, IMOD_OBJFLAG_OUT, IMOD_OBJFLAG_SCAT,
    IMOD_STRSIZE, IMOD_UNIT_MM, IMOD_UNIT_NM, IMOD_UNIT_UM, IMODF_FLIPYZ, IMODF_OTRANS_ORIGIN,
    IMODF_TILTOK, IOBJ_STRSIZE, Iclip_planes, Icont, Imesh, Imod, Iobj, Iobjview, Ipoint,
    Iref_image, Iview, Slicer_angles, imod_clean_surf,
};

/// Original: `IMODF_HAS_MESH_THICK` (`imodel.h:122`).
pub const IMODF_HAS_MESH_THICK: u32 = 1 << 9;
/// Original: `IMODF_MAT1_IS_BYTES` (`imodel.h:118`).
pub const IMODF_MAT1_IS_BYTES: u32 = 1 << 13;
use super::icont::imod_contours_new;
use super::imesh::imod_meshes_new;
use super::imodel_from::substr;
use super::iobj::{
    IMOD_OBJFLAG_ANTI_ALIAS, IMOD_OBJFLAG_FCOLOR, IMOD_OBJFLAG_FCOLOR_PNT, IMOD_OBJFLAG_FILL,
    IMOD_OBJFLAG_MCOLOR, IMOD_OBJFLAG_MESH, IMOD_OBJFLAG_NOLINE, IMOD_OBJFLAG_PNT_ON_SEC,
    IMOD_OBJFLAG_TIME, IMOD_OBJFLAG_TWO_SIDE, IMOD_OBJFLAG_USE_VALUE, imod_objects_new,
};
use super::ipoint::imod_point_set_size;
use super::istore::{
    Istore, imod_read_store, imod_write_store, istore_add_min_max, istore_end_change,
    istore_insert, istore_insert_change,
};
use super::iview::{VIEW_STRSIZE, imod_imnx_new, imod_view_model_new};
use crate::imod::libcfshr::b3dutil::b3d_error;

use super::objgroup::{obj_group_list_write, obj_group_read};
use crate::imod::libcfshr::b3dutil::{CArg, ImodFile, c_format, c_format_bytes};

const ID_IMOD: u32 = u32::from_be_bytes(*b"IMOD");
const IMOD_V12: u32 = u32::from_be_bytes(*b"V1.2");
const ID_OBJT: u32 = u32::from_be_bytes(*b"OBJT");
const ID_CONT: u32 = u32::from_be_bytes(*b"CONT");
const ID_MESH: u32 = u32::from_be_bytes(*b"MESH");
const ID_SIZE: u32 = u32::from_be_bytes(*b"SIZE");
const ID_CLIP: u32 = u32::from_be_bytes(*b"CLIP");
const ID_IMAT: u32 = u32::from_be_bytes(*b"IMAT");
const ID_VIEW: u32 = u32::from_be_bytes(*b"VIEW");
const ID_MCLP: u32 = u32::from_be_bytes(*b"MCLP");
const ID_OGRP: u32 = u32::from_be_bytes(*b"OGRP");
const ID_MOST: u32 = u32::from_be_bytes(*b"MOST");
const ID_OBST: u32 = u32::from_be_bytes(*b"OBST");
const ID_COST: u32 = u32::from_be_bytes(*b"COST");
const ID_MEST: u32 = u32::from_be_bytes(*b"MEST");
const ID_PNTS: u32 = u32::from_be_bytes(*b"PNTS");
const ID_SLAN: u32 = u32::from_be_bytes(*b"SLAN");
const ID_MEPA: u32 = u32::from_be_bytes(*b"MEPA");
const ID_SKLI: u32 = u32::from_be_bytes(*b"SKLI");
const ID_OLBL: u32 = u32::from_be_bytes(*b"OLBL");
const ID_LABL: u32 = u32::from_be_bytes(*b"LABL");
/// Original: `IMOD_01` (`imodel.h:69`).
const IMOD_01: u32 = u32::from_be_bytes(*b"V0.1");
/// Original: `SIZE_SLAN` (`imodel.h:100`).
const SIZE_SLAN: i32 = 60;
/// Original: `ANGLE_STRSIZE` (`imodel.h:35`).
const ANGLE_STRSIZE: usize = 32;
/// Original: `MAXLINE` (`imodel_files.c:28`).
const MAXLINE: i32 = 81;

/// Original: `writeAsciiClips` (`imodel_files.c:1834`).
pub fn write_ascii_clips(file: &mut ImodFile, clips: &super::imodel::Iclip_planes, prefix: &str) {
    if clips.count != 0 {
        {
            let _ = file.write_all(
                c_format(
                    "%s %d %d %d %d\n",
                    &[
                        CArg::Str(prefix),
                        CArg::Int(clips.count as i64),
                        CArg::Int(clips.flags as i64),
                        CArg::Int(clips.trans as i64),
                        CArg::Int(clips.plane as i64),
                    ],
                )
                .as_bytes(),
            );
            // `IMOD_CLIPSIZE` bounds the source arrays.
            for i in 0..(clips.count as usize).min(clips.normal.len()) {
                let _ = file.write_all(
                    c_format(
                        "%g %g %g %g %g %g\n",
                        &[
                            CArg::Dbl(clips.normal[i].x as f64),
                            CArg::Dbl(clips.normal[i].y as f64),
                            CArg::Dbl(clips.normal[i].z as f64),
                            CArg::Dbl(clips.point[i].x as f64),
                            CArg::Dbl(clips.point[i].y as f64),
                            CArg::Dbl(clips.point[i].z as f64),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
    }
}

/// Original: `imodWriteAscii` (`imodel_files.c:1631`).
pub fn imod_write_ascii(imod: &Imod, file: &mut ImodFile) -> i32 {
    {
        // `imodel_files.c:1645` `rewind(imod->file)`.
        let _ = file.rewind();
        let _ = file.write_all(b"# imod ascii file version 2.0\n\n");
        let _ =
            file.write_all(c_format("imod %d\n", &[CArg::Int(imod.obj.len() as i64)]).as_bytes());
        let _ = file.write_all(
            c_format(
                "max %d %d %d\n",
                &[
                    CArg::Int(imod.xmax as i64),
                    CArg::Int(imod.ymax as i64),
                    CArg::Int(imod.zmax as i64),
                ],
            )
            .as_bytes(),
        );
        let _ = file.write_all(
            c_format(
                "offsets %g %g %g\n",
                &[
                    CArg::Dbl(imod.xoffset as f64),
                    CArg::Dbl(imod.yoffset as f64),
                    CArg::Dbl(imod.zoffset as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = file.write_all(
            c_format(
                "angles %g %g %g\n",
                &[
                    CArg::Dbl(imod.alpha as f64),
                    CArg::Dbl(imod.beta as f64),
                    CArg::Dbl(imod.gamma as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = file.write_all(
            c_format(
                "scale %g %g %g\n",
                &[
                    CArg::Dbl(imod.xscale as f64),
                    CArg::Dbl(imod.yscale as f64),
                    CArg::Dbl(imod.zscale as f64),
                ],
            )
            .as_bytes(),
        );
        let _ = file
            .write_all(c_format("mousemode  %d\n", &[CArg::Int(imod.mousemode as i64)]).as_bytes());
        let _ = file
            .write_all(c_format("drawmode   %d\n", &[CArg::Int(imod.drawmode as i64)]).as_bytes());
        let _ = file.write_all(
            c_format(
                "b&w_level  %d,%d\n",
                &[
                    CArg::Int(imod.blacklevel as i64),
                    CArg::Int(imod.whitelevel as i64),
                ],
            )
            .as_bytes(),
        );
        let _ =
            file.write_all(c_format("resolution %d\n", &[CArg::Int(imod.res as i64)]).as_bytes());
        let _ = file
            .write_all(c_format("threshold  %d\n", &[CArg::Int(imod.thresh as i64)]).as_bytes());
        let _ = file
            .write_all(c_format("pixsize    %g\n", &[CArg::Dbl(imod.pixsize as f64)]).as_bytes());
        // `imodUnits` (`imodel.c:1360`) selects the name from the IMOD_UNIT_*
        // codes in `imodel.h:39-47`.
        let _ = file.write_all(
            c_format(
                "units      %s\n",
                &[CArg::Str(match imod.units {
                    0 => "pixels",
                    3 => "km",
                    1 => "m",
                    -2 => "cm",
                    -3 => "mm",
                    -6 => "um",
                    -9 => "nm",
                    -10 => "A",
                    -12 => "pm",
                    _ => "unknown units",
                })],
            )
            .as_bytes(),
        );
        let _ = file.write_all(
            c_format(
                "flipped    %d\n",
                &[CArg::Int(
                    (if imod.flags & super::imodel::IMODF_FLIPYZ != 0 {
                        1
                    } else {
                        0
                    }) as i64,
                )],
            )
            .as_bytes(),
        );
        if let Some(reference) = imod.ref_image {
            let _ = file.write_all(
                c_format(
                    "refcurscale %g %g %g\n",
                    &[
                        CArg::Dbl(reference.cscale.x as f64),
                        CArg::Dbl(reference.cscale.y as f64),
                        CArg::Dbl(reference.cscale.z as f64),
                    ],
                )
                .as_bytes(),
            );
            let _ = file.write_all(
                c_format(
                    "refcurtrans %g %g %g\n",
                    &[
                        CArg::Dbl(reference.ctrans.x as f64),
                        CArg::Dbl(reference.ctrans.y as f64),
                        CArg::Dbl(reference.ctrans.z as f64),
                    ],
                )
                .as_bytes(),
            );
            if imod.flags & super::imodel::IMODF_TILTOK != 0 {
                let _ = file.write_all(
                    c_format(
                        "refcurrot %g %g %g\n",
                        &[
                            CArg::Dbl(reference.crot.x as f64),
                            CArg::Dbl(reference.crot.y as f64),
                            CArg::Dbl(reference.crot.z as f64),
                        ],
                    )
                    .as_bytes(),
                );
            }
            if imod.flags & super::imodel::IMODF_OTRANS_ORIGIN != 0 {
                let _ = file.write_all(
                    c_format(
                        "refoldtrans %g %g %g\n",
                        &[
                            CArg::Dbl(reference.otrans.x as f64),
                            CArg::Dbl(reference.otrans.y as f64),
                            CArg::Dbl(reference.otrans.z as f64),
                        ],
                    )
                    .as_bytes(),
                );
            }
        }
        for angle in &imod.slicer_ang {
            // C `%s` copies the bytes of `slanp->label` through unchanged;
            // `String::from_utf8_lossy` would substitute U+FFFD for any that
            // are not UTF-8, so the raw bytes go to `CArg::Bytes`.
            let label_text = &angle.label[..angle
                .label
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(angle.label.len())];
            let _ = file.write_all(&c_format_bytes(
                "slicerAngle %d %g %g %g %g %g %g %s\n",
                &[
                    CArg::Int(angle.time as i64),
                    CArg::Dbl(angle.angles[0] as f64),
                    CArg::Dbl(angle.angles[1] as f64),
                    CArg::Dbl(angle.angles[2] as f64),
                    CArg::Dbl(angle.center.x as f64),
                    CArg::Dbl(angle.center.y as f64),
                    CArg::Dbl(angle.center.z as f64),
                    CArg::Bytes(label_text),
                ],
            ));
        }
        let _ = file
            .write_all(c_format("currentview  %d\n", &[CArg::Int(imod.cview as i64)]).as_bytes());
        for (iv, view) in imod.view.iter().enumerate().skip(1) {
            let _ = file.write_all(c_format("view %d\n", &[CArg::Int(iv as i64)]).as_bytes());
            let _ = file
                .write_all(c_format("viewfovy  %g\n", &[CArg::Dbl(view.fovy as f64)]).as_bytes());
            let _ = file
                .write_all(c_format("viewcnear %g\n", &[CArg::Dbl(view.cnear as f64)]).as_bytes());
            let _ = file
                .write_all(c_format("viewcfar  %g\n", &[CArg::Dbl(view.cfar as f64)]).as_bytes());
            let _ = file
                .write_all(c_format("viewflags %u\n", &[CArg::Uint(view.world as u64)]).as_bytes());
            let _ = file.write_all(
                c_format(
                    "viewtrans %g %g %g\n",
                    &[
                        CArg::Dbl(view.trans.x as f64),
                        CArg::Dbl(view.trans.y as f64),
                        CArg::Dbl(view.trans.z as f64),
                    ],
                )
                .as_bytes(),
            );
            let _ = file.write_all(
                c_format(
                    "viewrot %g %g %g\n",
                    &[
                        CArg::Dbl(view.rot.x as f64),
                        CArg::Dbl(view.rot.y as f64),
                        CArg::Dbl(view.rot.z as f64),
                    ],
                )
                .as_bytes(),
            );
            let _ = file.write_all(
                c_format(
                    "viewlight %g %g\n",
                    &[CArg::Dbl(view.lightx as f64), CArg::Dbl(view.lighty as f64)],
                )
                .as_bytes(),
            );
            let _ = file.write_all(
                c_format(
                    "depthcue %g %g\n",
                    &[CArg::Dbl(view.dcstart as f64), CArg::Dbl(view.dcend as f64)],
                )
                .as_bytes(),
            );
            // C `%s` copies the bytes of `view->label` through unchanged.
            let label_text = &view.label[..view
                .label
                .iter()
                .position(|byte| *byte == 0)
                .unwrap_or(view.label.len())];
            let _ = file.write_all(&c_format_bytes(
                "viewlabel %s\n",
                &[CArg::Bytes(label_text)],
            ));
            write_ascii_clips(file, &view.clips, "globalclips");
        }
        for (ob, obj) in imod.obj.iter().enumerate() {
            let num_real = obj
                .mesh
                .iter()
                .filter(|mesh| (mesh.flag >> 24) & 0x3f == 0)
                .count();
            let _ = file.write_all(
                c_format(
                    "\nobject %d %d %d\n",
                    &[
                        CArg::Int(ob as i64),
                        CArg::Int(obj.cont.len() as i64),
                        CArg::Int(num_real as i64),
                    ],
                )
                .as_bytes(),
            );
            let _ = file.write_all(&c_format_bytes(
                "name %s\n",
                &[CArg::Bytes(
                    &obj.name[..obj
                        .name
                        .iter()
                        .position(|byte| *byte == 0)
                        .unwrap_or(IOBJ_STRSIZE)],
                )],
            ));
            let _ = file.write_all(
                c_format(
                    "color %g %g %g %d\n",
                    &[
                        CArg::Dbl(obj.red as f64),
                        CArg::Dbl(obj.green as f64),
                        CArg::Dbl(obj.blue as f64),
                        CArg::Int(obj.trans as i64),
                    ],
                )
                .as_bytes(),
            );
            if obj.fillred != 0 || obj.fillgreen != 0 || obj.fillblue != 0 {
                let _ = file.write_all(
                    c_format(
                        "Fillcolor %d %d %d\n",
                        &[
                            CArg::Int(obj.fillred as i64),
                            CArg::Int(obj.fillgreen as i64),
                            CArg::Int(obj.fillblue as i64),
                        ],
                    )
                    .as_bytes(),
                );
            }
            for (flag, line) in [
                (super::imodel::IMOD_OBJFLAG_OPEN, "open\n"),
                (super::imodel::IMOD_OBJFLAG_SCAT, "scattered\n"),
                (super::imodel::IMOD_OBJFLAG_OFF, "nodraw\n"),
                (super::imodel::IMOD_OBJFLAG_OUT, "insideout\n"),
                (1 << 8, "fill\n"),
                (1 << 10, "drawmesh\n"),
                (1 << 11, "nolines\n"),
                (1 << 19, "bothsides\n"),
                (1 << 14, "usefill\n"),
                (1 << 6, "pntusefill\n"),
                (1 << 7, "pntonsec\n"),
                (1 << 15, "antialias\n"),
                (1 << 18, "hastimes\n"),
                (1 << 12, "usevalue\n"),
                (1 << 17, "valcolor\n"),
            ] {
                if obj.flags & flag != 0 {
                    let _ = file.write_all(line.as_bytes());
                }
            }
            let _ = file.write_all(
                c_format("linewidth %d\n", &[CArg::Int(obj.linewidth as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("surfsize  %d\n", &[CArg::Int(obj.surfsize as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("pointsize %d\n", &[CArg::Int(obj.pdrawsize as i64)]).as_bytes(),
            );
            let _ = file
                .write_all(c_format("axis      %d\n", &[CArg::Int(obj.axis as i64)]).as_bytes());
            let _ = file.write_all(
                c_format("drawmode  %d\n", &[CArg::Int(obj.drawmode as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("width2D   %d\n", &[CArg::Int(obj.linewidth2 as i64)]).as_bytes(),
            );
            let _ = file
                .write_all(c_format("symbol    %d\n", &[CArg::Int(obj.symbol as i64)]).as_bytes());
            let _ = file
                .write_all(c_format("symsize   %d\n", &[CArg::Int(obj.symsize as i64)]).as_bytes());
            let _ = file.write_all(
                c_format("symflags  %d\n", &[CArg::Int(obj.symflags as i64)]).as_bytes(),
            );
            let _ = file
                .write_all(c_format("ambient   %d\n", &[CArg::Int(obj.ambient as i64)]).as_bytes());
            let _ = file
                .write_all(c_format("diffuse   %d\n", &[CArg::Int(obj.diffuse as i64)]).as_bytes());
            let _ = file.write_all(
                c_format("specular  %d\n", &[CArg::Int(obj.specular as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("shininess %d\n", &[CArg::Int(obj.shininess as i64)]).as_bytes(),
            );
            let _ = file
                .write_all(c_format("obquality %d\n", &[CArg::Int(obj.quality as i64)]).as_bytes());
            let _ = file.write_all(
                c_format("valblack  %d\n", &[CArg::Int(obj.valblack as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("valwhite  %d\n", &[CArg::Int(obj.valwhite as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("meshthick %d\n", &[CArg::Int(obj.mesh_thickness as i64)]).as_bytes(),
            );
            let _ = file.write_all(
                c_format("matflags2 %d\n", &[CArg::Int(obj.matflags2 as i64)]).as_bytes(),
            );
            write_ascii_clips(file, &obj.clips, "objclips");
            let mut valmin = 0.;
            let mut valmax = 0.;
            if super::istore::istore_get_min_max(
                &obj.store,
                obj.cont.len() as i32,
                11,
                &mut valmin,
                &mut valmax,
            ) != 0
            {
                let _ = file.write_all(
                    c_format(
                        "valminmax %g %g\n",
                        &[CArg::Dbl(valmin as f64), CArg::Dbl(valmax as f64)],
                    )
                    .as_bytes(),
                );
            }
            let mut def_props = super::istore::DrawProps::default();
            super::istore::istore_default_draw_props(obj, &mut def_props);
            for (co, cont) in obj.cont.iter().enumerate() {
                let mut cont_props = super::istore::DrawProps::default();
                let mut cont_state = 0;
                let mut surf_state = 0;
                super::istore::istore_cont_surf_draw_props(
                    &obj.store,
                    &def_props,
                    &mut cont_props,
                    co as i32,
                    cont.surf,
                    &mut cont_state,
                    &mut surf_state,
                );
                let _ = file.write_all(
                    c_format(
                        "contour %d %d %d",
                        &[
                            CArg::Int(co as i64),
                            CArg::Int(cont.surf as i64),
                            CArg::Int(cont.pts.len() as i64),
                        ],
                    )
                    .as_bytes(),
                );
                if cont_state & (1 << 9) != 0 {
                    let _ = file.write_all(
                        c_format(" %g", &[CArg::Dbl(cont_props.value1 as f64)]).as_bytes(),
                    );
                }
                let _ = file.write_all(b"\n");
                let mut cursor = 0;
                let mut state = 0;
                let mut changes = 0;
                let mut props = super::istore::DrawProps::default();
                let mut next = super::istore::istore_first_change_index(&cont.store);
                for (pt, point) in cont.pts.iter().enumerate() {
                    let _ = file.write_all(
                        c_format(
                            "%g %g %g",
                            &[
                                CArg::Dbl(point.x as f64),
                                CArg::Dbl(point.y as f64),
                                CArg::Dbl(point.z as f64),
                            ],
                        )
                        .as_bytes(),
                    );
                    if pt as i32 == next {
                        next = super::istore::istore_next_change(
                            &cont.store,
                            &mut cursor,
                            &cont_props,
                            &mut props,
                            &mut state,
                            &mut changes,
                        );
                    }
                    let mut size = -1.0_f32;
                    if cont.sizes.get(pt).is_some_and(|value| *value >= 0.) {
                        size = cont.sizes[pt];
                    }
                    if size >= 0. && state & (1 << 9) == 0 {
                        let _ =
                            file.write_all(c_format(" %g", &[CArg::Dbl(size as f64)]).as_bytes());
                    } else if state & (1 << 9) != 0 {
                        let _ = file.write_all(
                            c_format(
                                " %g %g",
                                &[CArg::Dbl(size as f64), CArg::Dbl(props.value1 as f64)],
                            )
                            .as_bytes(),
                        );
                    }
                    let _ = file.write_all(b"\n");
                }
                if cont.flags != 0 {
                    let _ = file.write_all(
                        c_format("contflags %u\n", &[CArg::Uint(cont.flags as u64)]).as_bytes(),
                    );
                }
                if cont.time != 0 {
                    let _ = file.write_all(
                        c_format("conttime %d\n", &[CArg::Int(cont.time as i64)]).as_bytes(),
                    );
                }
            }
            let mut num_real = 0;
            for mesh in &obj.mesh {
                if (mesh.flag >> 24) & 0x3f != 0 {
                    continue;
                }
                let _ = file.write_all(
                    c_format(
                        "mesh %d %d %d\n",
                        &[
                            CArg::Int(num_real as i64),
                            CArg::Int(mesh.vert.len() as i64),
                            CArg::Int(mesh.list.len() as i64),
                        ],
                    )
                    .as_bytes(),
                );
                num_real += 1;
                for point in &mesh.vert {
                    let _ = file.write_all(
                        c_format(
                            "%g %g %g\n",
                            &[
                                CArg::Dbl(point.x as f64),
                                CArg::Dbl(point.y as f64),
                                CArg::Dbl(point.z as f64),
                            ],
                        )
                        .as_bytes(),
                    );
                }
                for index in &mesh.list {
                    let _ =
                        file.write_all(c_format("%d\n", &[CArg::Int((*index) as i64)]).as_bytes());
                }
                if mesh.flag != 0 {
                    let _ = file.write_all(
                        c_format("Meshflags %u\n", &[CArg::Uint(mesh.flag as u64)]).as_bytes(),
                    );
                }
                if mesh.time != 0 {
                    let _ = file.write_all(
                        c_format("Meshtime %d\n", &[CArg::Int(mesh.time as i64)]).as_bytes(),
                    );
                }
                if mesh.surf != 0 {
                    let _ = file.write_all(
                        c_format("Meshsurf %d\n", &[CArg::Int(mesh.surf as i64)]).as_bytes(),
                    );
                }
            }
        }
        let _ = file.write_all(b"# end of IMOD model\n");
    }
    0
}
const ID_IMNX: u32 = u32::from_be_bytes(*b"MINX");
const ID_IEOF: u32 = u32::from_be_bytes(*b"IEOF");

/// Original: `imodGetInt` (`imodel_files.c:2074`).
pub fn imod_get_int(file: &mut ImodFile) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_be_bytes(bytes))
}

/// Original: `imodGetFloat` (`imodel_files.c:2018`).
pub fn imod_get_float(file: &mut ImodFile) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(f32::from_bits(u32::from_be_bytes(bytes)))
}

/// Original: `imodGetShort` (`imodel_files.c:2106`).
pub fn imod_get_short(file: &mut ImodFile) -> io::Result<i16> {
    let mut bytes = [0_u8; 2];
    file.read_exact(&mut bytes)?;
    Ok(i16::from_be_bytes(bytes))
}

/// Original: `imodGetByte` (`imodel_files.c:2129`).
pub fn imod_get_byte(file: &mut ImodFile) -> io::Result<u8> {
    let mut bytes = [0_u8; 1];
    file.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

/// Original: `imodPutInt` (`imodel_files.c:2090`).
pub fn imod_put_int(file: &mut ImodFile, value: i32) -> io::Result<()> {
    file.write_all(&value.to_be_bytes())
}

/// Original: `imodPutFloat` (`imodel_files.c:2039`).
pub fn imod_put_float(file: &mut ImodFile, value: f32) -> io::Result<()> {
    file.write_all(&value.to_bits().to_be_bytes())
}

/// Original: `imodPutShort` (`imodel_files.c:2115`).
pub fn imod_put_short(file: &mut ImodFile, value: i16) -> io::Result<()> {
    file.write_all(&value.to_be_bytes())
}

/// Original: `imodPutByte` (`imodel_files.c:2146`).
pub fn imod_put_byte(file: &mut ImodFile, value: u8) -> io::Result<()> {
    file.write_all(&[value])
}

/// Original: `byteswap` (`imodel_files.c:1893`).
///
/// Compiled only under `IMOD_DATA_SWAP`, which `imodel_files.c:25` defines for
/// every little-endian host.
pub fn byteswap(iptr: &mut [u8], mut size: u32) {
    if (size % 2) != 0 {
        size -= 1;
    }

    let mut begin = 0usize;
    let mut end = size.wrapping_sub(1) as usize;

    for _ in 0..(size / 2) {
        iptr.swap(begin, end);

        begin += 1;
        end -= 1;
    }
}

/// Original: `swap_longs` (`imodel_files.c:1918`).
pub fn swap_longs(data: &mut [u8], amt: i32) {
    let ldata = (amt * 4) as usize;
    let mut ptr = 0usize;
    while ptr < ldata {
        data.swap(ptr, ptr + 3);
        ptr += 1;
        data.swap(ptr, ptr + 1);
        ptr += 3;
    }
}

/// Original: `BUF_SIZE` (`imodel_files.c:1934`).
const BUF_SIZE: usize = 256;

/// Original: `convert_write` (`imodel_files.c:1938`).
///
/// The source's scratch array is a file-level `static int buf[BUF_SIZE]`; a
/// local array of the same size is used here.
pub fn convert_write(
    convert_func: fn(&mut [u8], i32),
    fp: &mut ImodFile,
    data: &[u8],
    mut size: i32,
) {
    let mut buf = [0u8; 4 * BUF_SIZE];
    let mut chunk;
    let mut base = 0usize;
    while size != 0 {
        chunk = if (size as usize) < BUF_SIZE {
            size as usize
        } else {
            BUF_SIZE
        };
        buf[..4 * chunk].copy_from_slice(&data[base..base + 4 * chunk]);
        convert_func(&mut buf, chunk as i32);
        let _ = fp.write_all(&buf[..4 * chunk]);
        size -= chunk as i32;
        base += 4 * chunk;
    }
}

/// Original: `tovmsfloat` (`imodel_files.c:1956`).
///
/// Compiled only under `IMOD_FLOAT_CONVERT`, which `imodel_files.c:20` defines
/// for VMS hosts alone; it is never reached on this platform.
pub fn tovmsfloat(data: &mut [u8], amt: i32) {
    let mut ptr = 0usize;
    let maxptr = (amt * 4) as usize;

    while ptr < maxptr {
        let exp = (data[ptr] << 1) | (data[ptr + 1] >> 7 & 0x01);
        if exp < 253 && exp != 0 {
            data[ptr] = data[ptr].wrapping_add(1);
        } else if exp >= 253
        /*must also max out the exp & mantissa*/
        {
            /*we want manitssa all 1 & exponent 255*/
            data[ptr] |= 0x7F;
            data[ptr + 1] = 0xFF;
            data[ptr + 3] = 0xFF;
            data[ptr + 2] = data[ptr + 3];
        }

        data.swap(ptr, ptr + 1);
        data.swap(ptr + 2, ptr + 3);
        ptr += 4;
    }
}

/// Original: `imodFromVmsFloats` (`imodel_files.c:1987`).
pub fn imod_from_vms_floats(data: &mut [u8], amt: i32) {
    let mut ptr = 0usize;
    let maxptr = (amt * 4) as usize;

    while ptr < maxptr {
        let exp = (data[ptr + 1] << 1) | (data[ptr] >> 7 & 0x01);
        if exp > 3 && exp != 0 {
            data[ptr + 1] = data[ptr + 1].wrapping_sub(1);
        } else if exp <= 3 && exp != 0 {
            /*we want manitssa 0 & exponent 1*/
            data[ptr] = 0x80;
            data[ptr + 1] &= 0x80;
            data[ptr + 3] = 0;
            data[ptr + 2] = data[ptr + 3];
        }

        data.swap(ptr, ptr + 1);
        data.swap(ptr + 2, ptr + 3);
        ptr += 4;
    }
}

/// Original: `imodGetFloats` (`imodel_files.c:2025`).
pub fn imod_get_floats(fp: &mut ImodFile, buf: &mut [f32], size: i32) -> io::Result<()> {
    let mut bytes = vec![0u8; 4 * size.max(0) as usize];
    fp.read_exact(&mut bytes)?;
    swap_longs(&mut bytes, size);
    for i in 0..size.max(0) as usize {
        buf[i] = f32::from_ne_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]);
    }
    Ok(())
}

/// Original: `imodPutFloats` (`imodel_files.c:2044`).
pub fn imod_put_floats(fp: &mut ImodFile, buf: &[f32], size: i32) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(4 * size.max(0) as usize);
    for value in buf.iter().take(size.max(0) as usize) {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    convert_write(swap_longs, fp, &bytes, size);
    Ok(())
}

/// Original: `imodPutScaledPoints` (`imodel_files.c:2058`).
pub fn imod_put_scaled_points(
    fp: &mut ImodFile,
    buf: &[Ipoint],
    size: i32,
    scale: &Ipoint,
) -> io::Result<()> {
    let mut xyz = [0f32; 3];
    for i in 0..size.max(0) as usize {
        xyz[0] = buf[i].x * scale.x;
        xyz[1] = buf[i].y * scale.y;
        xyz[2] = buf[i].z * scale.z;
        imod_put_floats(fp, &xyz, 3)?;
    }
    Ok(())
}

/// Original: `imodGetInts` (`imodel_files.c:2081`).
pub fn imod_get_ints(fp: &mut ImodFile, buf: &mut [i32], size: i32) -> io::Result<()> {
    let mut bytes = vec![0u8; 4 * size.max(0) as usize];
    fp.read_exact(&mut bytes)?;
    swap_longs(&mut bytes, size);
    for i in 0..size.max(0) as usize {
        buf[i] = i32::from_ne_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]);
    }
    Ok(())
}

/// Original: `imodPutInts` (`imodel_files.c:2095`).
pub fn imod_put_ints(fp: &mut ImodFile, buf: &[i32], size: i32) -> io::Result<()> {
    let mut bytes = Vec::with_capacity(4 * size.max(0) as usize);
    for value in buf.iter().take(size.max(0) as usize) {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    convert_write(swap_longs, fp, &bytes, size);
    Ok(())
}

/// Original: `imodGetBytes` (`imodel_files.c:2135`).
pub fn imod_get_bytes(fp: &mut ImodFile, buf: &mut [u8], size: i32) -> io::Result<()> {
    fp.read_exact(&mut buf[..size.max(0) as usize])
}

/// Original: `imodPutBytes` (`imodel_files.c:2141`).
pub fn imod_put_bytes(fp: &mut ImodFile, buf: &[u8], size: i32) -> io::Result<()> {
    fp.write_all(&buf[..size.max(0) as usize])
}

/// Original: `imodFgetline` (`imodel_files.c:1847`).
pub fn imod_fgetline(fp: &mut ImodFile, s: &mut [u8], limit: i32) -> i32 {
    let mut c: i32;
    let i: i32;

    if limit < 3 {
        return -1;
    }

    let mut idx = 0usize;
    loop {
        let mut byte = [0u8; 1];
        c = match fp.read(&mut byte) {
            Ok(1) => byte[0] as i32,
            _ => -1,
        };
        if c != b'\r' as i32 {
            s[idx] = c as u8;
            idx += 1;
        }
        if !((c != -1) && ((idx as i32) < (limit - 1)) && (c != b'\n' as i32)) {
            break;
        }
    }
    i = idx as i32;

    if i == 1 {
        if c == -1 {
            return 0;
        }
        if c == b'\n' as i32 {
            idx += 1;
            s[idx] = b'\0';
            return imod_fgetline(fp, s, limit);
        }
    }

    if s[0] == b'#' {
        return imod_fgetline(fp, s, limit);
    }

    s[idx] = b'\0';
    let length = idx as i32;

    if c == -1 { -1 * length } else { length }
}

/// Original: `readAsciiClips` (`imodel_files.c:1611`).
///
/// The source reads the four counts with `sscanf` into uninitialised locals, so
/// a short line leaves them holding stack garbage; they are zeroed here.  The
/// `imod->file` argument is passed explicitly because the translated `Imod`
/// carries no `file` member.
pub fn read_ascii_clips(
    fp: &mut ImodFile,
    line: &mut [u8],
    clips: &mut Iclip_planes,
    prefix: &str,
) {
    let mut idata = 0i32;
    let mut idata2 = 0i32;
    let mut idata3 = 0i32;
    let mut idata4 = 0i32;
    let text = String::from_utf8_lossy(&line[prefix.len()..]).to_string();
    let mut fields = text
        .split('\0')
        .next()
        .unwrap_or("")
        .split_whitespace()
        .map(|word| word.parse::<i32>().ok());
    for slot in [&mut idata, &mut idata2, &mut idata3, &mut idata4] {
        match fields.next().flatten() {
            Some(value) => *slot = value,
            None => break,
        }
    }
    clips.count = idata as u8;
    clips.flags = idata2 as u8;
    clips.trans = idata3 as u8;
    clips.plane = idata4 as u8;
    // `IMOD_CLIPSIZE` bounds the source arrays.
    for i in 0..(clips.count as usize).min(clips.normal.len()) {
        imod_fgetline(fp, line, MAXLINE);
        let text = String::from_utf8_lossy(line).to_string();
        let values: Vec<f32> = text
            .split('\0')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .map(|word| word.parse::<f32>().unwrap_or(0.))
            .collect();
        for (index, value) in values.iter().enumerate().take(6) {
            match index {
                0 => clips.normal[i].x = *value,
                1 => clips.normal[i].y = *value,
                2 => clips.normal[i].z = *value,
                3 => clips.point[i].x = *value,
                4 => clips.point[i].y = *value,
                _ => clips.point[i].z = *value,
            }
        }
    }
}

/// Original: `imodel_read_header` (`imodel_files.c:835`).
///
/// The translated `Imod` has no `objsize` member — the object array is the
/// `obj` vector — so the object count read from the header is returned to the
/// caller instead of being stored.
pub fn imodel_read_header(imod: &mut Imod, file: &mut ImodFile) -> Result<i32, i32> {
    // `imodel_files.c:837` reads IMOD_STRSIZE bytes into the fixed `char name[]`
    // array and keeps every one of them; `%s` output stops at the first NUL but
    // the bytes past it are written back out verbatim by `imodel_write`.
    file.read_exact(&mut imod.name)
        .map_err(|_| IMOD_ERROR_READ)?;
    imod.xmax = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.ymax = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.zmax = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let objsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.obj.reserve(objsize.max(0) as usize);
    imod.flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    imod.drawmode = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.mousemode = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.blacklevel = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.whitelevel = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.xoffset = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.yoffset = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.zoffset = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.xscale = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.yscale = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.zscale = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.cindex.object = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.cindex.contour = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.cindex.point = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.res = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.thresh = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.pixsize = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.units = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    // `imodel_files.c:845` stores the checksum field; `imodel_files.c:307`
    // writes it back out unchanged.
    imod.csum = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.alpha = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.beta = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    imod.gamma = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    Ok(objsize)
}

/// Original: `imodel_read_object` (`imodel_files.c:860`).
///
/// The translated `Iobj` has no `contsize`/`meshsize` members — the contour and
/// mesh arrays are vectors — so those two counts are returned to the caller,
/// which allocates the arrays exactly as `imodel_read` does in the source.
pub fn imodel_read_object(obj: &mut Iobj, file: &mut ImodFile) -> Result<(i32, i32), i32> {
    // `imodel_files.c:862` reads IOBJ_STRSIZE bytes into `char name[]` and keeps
    // all of them; `%s` output stops at the first NUL but `imodel_write_object`
    // writes the whole array back out.
    file.read_exact(&mut obj.name)
        .map_err(|_| IMOD_ERROR_READ)?;
    for value in &mut obj.extra {
        *value = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    }
    let contsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    obj.axis = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.drawmode = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.red = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.green = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.blue = imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.pdrawsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.symbol = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.symsize = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    if obj.symsize == 0 {
        obj.symsize = 3;
    }
    obj.linewidth2 = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    if obj.linewidth2 == 0 {
        obj.linewidth2 = 1;
    }
    obj.linewidth = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.linesty = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.symflags = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.sympad = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.trans = imod_get_byte(file).map_err(|_| IMOD_ERROR_READ)?;
    let meshsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.surfsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    Ok((contsize, meshsize))
}

/// Original: `imodel_read_object_v01` (`imodel_files.c:892`).
///
/// The source reads `IMOD_STRSIZE` (128) bytes into the object's 64-byte `name`
/// array, overrunning it into the following struct members, which the reads
/// that follow then overwrite.  Shipping that overrun is not possible here, so
/// the 128 bytes are read and only the first `IOBJ_STRSIZE` are kept.
///
/// Returns the contour count, which the translated `Iobj` does not carry.
pub fn imodel_read_object_v01(obj: &mut Iobj, file: &mut ImodFile) -> Result<i32, i32> {
    loop {
        /* allow for extra chunks after model data */
        let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        if id == ID_OBJT {
            break;
        }
        let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        if file.seek(SeekFrom::Current(id as i64)).is_err() {
            return Err(-1);
        }
    }

    let mut name = [0u8; IMOD_STRSIZE];
    imod_get_bytes(file, &mut name, IMOD_STRSIZE as i32).map_err(|_| IMOD_ERROR_READ)?;
    for i in 0..IOBJ_STRSIZE {
        obj.name[i] = name[i];
    }
    obj.name[IOBJ_STRSIZE - 1] = 0x00;
    for i in 0..obj.extra.len() {
        obj.extra[i] = 0;
    }

    let mut counts = [0i32; 4];
    imod_get_ints(file, &mut counts, 4).map_err(|_| IMOD_ERROR_READ)?;
    let contsize = counts[0];
    obj.flags = counts[1] as u32;
    obj.axis = counts[2];
    obj.drawmode = counts[3];
    let mut rgb = [0f32; 3];
    imod_get_floats(file, &mut rgb, 3).map_err(|_| IMOD_ERROR_READ)?;
    obj.red = rgb[0];
    obj.green = rgb[1];
    obj.blue = rgb[2];
    obj.pdrawsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;

    /* Combatibility with old imod files. */
    if obj.pdrawsize < 1 {
        obj.pdrawsize = 1;
    }

    let mut tmp = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.linewidth = tmp as u8;
    if obj.linewidth < 1 {
        obj.linewidth = 1;
    }
    obj.linewidth2 = 1;
    obj.symbol = 0;
    obj.symsize = 3;

    obj.linesty = 0;
    obj.symflags = 0;
    obj.sympad = 0;

    tmp = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    obj.trans = tmp as u8;

    let meshsize = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let _ = meshsize;

    if contsize > 0 {
        let Some(cont) = super::icont::imod_contours_new(contsize) else {
            return Err(IMOD_ERROR_MEMORY);
        };
        obj.cont = cont;
    }

    for i in 0..contsize.max(0) as usize {
        imodel_read_contour_v01(&mut obj.cont[i], file)?;
    }

    Ok(contsize)
}

/// Original: `imodel_read_contour_v01` (`imodel_files.c:952`).
pub fn imodel_read_contour_v01(cont: &mut Icont, file: &mut ImodFile) -> Result<(), i32> {
    loop {
        /* allow for extra chunks after object data */
        let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        if id == ID_CONT {
            break;
        }
        let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
        if file.seek(SeekFrom::Current(id as i64)).is_err() {
            return Err(-1);
        }
    }
    let mut head = [0i32; 4];
    imod_get_ints(file, &mut head, 4).map_err(|_| IMOD_ERROR_READ)?;
    let psize = head[0];
    cont.flags = head[1] as u32;
    cont.time = head[2];
    cont.surf = head[3];
    let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    if id != ID_PNTS {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("IMOD: Error Reading Points.\n"),
        );
        return Err(-1);
    }
    if psize < 0 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("IMOD: Error getting memory for points.\n"),
        );
        return Err(-1);
    }
    cont.pts = vec![Ipoint::default(); psize as usize];
    let mut xyz = vec![0f32; 3 * psize as usize];
    imod_get_floats(file, &mut xyz, psize * 3).map_err(|_| IMOD_ERROR_READ)?;
    for i in 0..psize as usize {
        cont.pts[i].x = xyz[3 * i];
        cont.pts[i].y = xyz[3 * i + 1];
        cont.pts[i].z = xyz[3 * i + 2];
    }
    Ok(())
}

/// Original: `imodel_read_contour` (`imodel_files.c:984`).
pub fn imodel_read_contour(cont: &mut Icont, file: &mut ImodFile) -> Result<(), i32> {
    let size = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    cont.flags = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    cont.time = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    cont.surf = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if size < 0 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    for _ in 0..size {
        cont.pts.push(Ipoint {
            x: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            y: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            z: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
        });
    }
    Ok(())
}

/// Original: `imodel_read_ptsizes` (`imodel_files.c:1022`).
pub fn imodel_read_ptsizes(cont: &mut Icont, file: &mut ImodFile) -> Result<(), i32> {
    let byte_count = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    if byte_count != (cont.pts.len() * 4) as i32 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    for _ in 0..cont.pts.len() {
        cont.sizes
            .push(imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?);
    }
    Ok(())
}

/// Original: `imodel_read_mesh` (`imodel_files.c:1043`).
pub fn imodel_read_mesh(mesh: &mut Imesh, file: &mut ImodFile) -> Result<(), i32> {
    let vertices = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let lists = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    mesh.flag = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    mesh.time = imod_get_short(file).map_err(|_| IMOD_ERROR_READ)?;
    mesh.surf = imod_get_short(file).map_err(|_| IMOD_ERROR_READ)?;
    if vertices < 0 || lists < 0 {
        return Err(IMOD_ERROR_CORRUPT);
    }
    for _ in 0..vertices {
        mesh.vert.push(Ipoint {
            x: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            y: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
            z: imod_get_float(file).map_err(|_| IMOD_ERROR_READ)?,
        });
    }
    for _ in 0..lists {
        mesh.list
            .push(imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?);
    }
    Ok(())
}

/// Original: `imodel_read_imat` (`imodel_files.c:1106`).
pub fn imodel_read_imat(object: &mut Iobj, file: &mut ImodFile, flags: u32) -> Result<(), i32> {
    let _ = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
    let mut value = [0; 4];
    file.read_exact(&mut value).map_err(|_| IMOD_ERROR_READ)?;
    object.ambient = value[0];
    object.diffuse = value[1];
    object.specular = value[2];
    object.shininess = value[3];
    if flags & (1 << 13) != 0 {
        file.read_exact(&mut value).map_err(|_| IMOD_ERROR_READ)?;
        object.fillred = value[0];
        object.fillgreen = value[1];
        object.fillblue = value[2];
        object.quality = value[3];
        object.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        file.read_exact(&mut value).map_err(|_| IMOD_ERROR_READ)?;
        object.valblack = value[0];
        object.valwhite = value[1];
        object.matflags2 = value[2];
        object.mesh_thickness = value[3];
    } else {
        let value = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        object.fillred = (value >> 24) as u8;
        object.fillgreen = (value >> 16) as u8;
        object.fillblue = (value >> 8) as u8;
        object.quality = value as u8;
        object.mat2 = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        let value = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        object.valblack = (value >> 24) as u8;
        object.valwhite = (value >> 16) as u8;
        object.matflags2 = (value >> 8) as u8;
    }
    if flags & (1 << 9) == 0 {
        object.mesh_thickness = 0;
    }
    Ok(())
}

/// Original: `imodel_read_clip` (`imodel_files.c:1099`).
///
/// The source delegates to `imodClipsRead` (`iplane.c:166`), which sizes the
/// read from the chunk length and reads all the normals before all the points,
/// then to `imodClipsFixCount` (`iplane.c:186`).  Its return value is the read
/// error, which `imodel_read` discards.
pub fn imodel_read_clip(object: &mut Iobj, file: &mut ImodFile, flags: u32) -> i32 {
    let error = crate::imod::libimod::iplane::imod_clips_read(&mut object.clips, file);
    crate::imod::libimod::iplane::imod_clips_fix_count(&mut object.clips, flags);
    error
}

/// Original: `imodel_read_meshparm` (`imodel_files.c:1126`).
///
pub fn imodel_read_meshparm(obj: &mut Iobj, fin: &mut ImodFile) -> Result<i32, i32> {
    let _ = imod_get_int(fin).map_err(|_| IMOD_ERROR_READ)?;
    let mut ints = [0i32; 9];
    imod_get_ints(fin, &mut ints, 9).map_err(|_| IMOD_ERROR_READ)?;
    let mut floats = [0f32; 10];
    imod_get_floats(fin, &mut floats, 10).map_err(|_| IMOD_ERROR_READ)?;
    obj.mesh_param = Some(crate::imod::libimod::imesh::MeshParams {
        flags: ints[0] as u32,
        cap: ints[1],
        passes: ints[2],
        cap_skip_nz: ints[3],
        incz_low_res: ints[4],
        incz_high_res: ints[5],
        minz: ints[6],
        maxz: ints[7],
        spare_int: ints[8],
        overlap: floats[0],
        tube_diameter: floats[1],
        xmin: floats[2],
        xmax: floats[3],
        ymin: floats[4],
        ymax: floats[5],
        tol_low_res: floats[6],
        tol_high_res: floats[7],
        flat_crit: floats[8],
        spare_float: floats[9],
        cap_skip_zlist: None,
    });
    Ok(ints[3])
}

/// Original: `imodel_read_meshskip` (`imodel_files.c:1140`).
///
pub fn imodel_read_meshskip(
    obj: &mut Iobj,
    fin: &mut ImodFile,
    cap_skip_nz: Option<i32>,
) -> Result<(), i32> {
    let size = imod_get_int(fin).map_err(|_| IMOD_ERROR_READ)? / 4;
    if cap_skip_nz != Some(size) {
        return Err(IMOD_ERROR_FORMAT);
    }
    let mut list = vec![0i32; size.max(0) as usize];
    imod_get_ints(fin, &mut list, size).map_err(|_| IMOD_ERROR_READ)?;
    if let Some(params) = obj.mesh_param.as_mut() {
        params.cap_skip_zlist = Some(list);
    }
    Ok(())
}

/// Original: `imodel_read_sliceang` (`imodel_files.c:1155`).
pub fn imodel_read_sliceang(imod: &mut Imod, fin: &mut ImodFile) -> Result<(), i32> {
    let mut slan = Slicer_angles::default();
    let _ = imod_get_int(fin).map_err(|_| IMOD_ERROR_READ)?;
    slan.time = imod_get_int(fin).map_err(|_| IMOD_ERROR_READ)?;
    let mut angles = [0f32; 6];
    imod_get_floats(fin, &mut angles, 6).map_err(|_| IMOD_ERROR_READ)?;
    slan.angles[0] = angles[0];
    slan.angles[1] = angles[1];
    slan.angles[2] = angles[2];
    slan.center.x = angles[3];
    slan.center.y = angles[4];
    slan.center.z = angles[5];
    imod_get_bytes(fin, &mut slan.label, ANGLE_STRSIZE as i32).map_err(|_| IMOD_ERROR_READ)?;
    imod.slicer_ang.push(slan);
    Ok(())
}

/// Original: `imodel_read_v01` (`imodel_files.c:512`).
pub fn imodel_read_v01(imod: &mut Imod, fin: &mut ImodFile) -> Result<(), i32> {
    fin.seek(SeekFrom::Start(0)).map_err(|_| IMOD_ERROR_READ)?;
    let id = imod_get_int(fin).map_err(|_| IMOD_ERROR_READ)? as u32;
    if id != ID_IMOD {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("Read Imod: Not an imod file.\n"),
        );
        return Err(-1);
    }
    let id = imod_get_int(fin).map_err(|_| IMOD_ERROR_READ)? as u32;
    if id != IMOD_01 {
        b3d_error(
            Some(&mut ImodFile::Stderr),
            format_args!("Read Imod: Imod file version unknown.\n"),
        );
        return Err(-1);
    }

    let objsize = imodel_read_header(imod, fin)?;

    imod.obj = super::iobj::imod_objects_new(objsize).ok_or(IMOD_ERROR_MEMORY)?;

    for i in 0..objsize.max(0) as usize {
        if imodel_read_object_v01(&mut imod.obj[i], fin).is_err() {
            return Err(-1);
        }
    }

    Ok(())
}

/// Original: `imodel_read` (`imodel_files.c:555`).
///
/// The source's loop is `while (!feof(fin) && !ieof)` and relies on a failed
/// `imodGetInt` leaving an indeterminate chunk id behind; a failed read of the
/// chunk id is reported as `IMOD_ERROR_READ` here instead.
pub fn imodel_read(imod: &mut Imod, file: &mut ImodFile, version: u32) -> Result<(), i32> {
    let mut obj_ind: i32 = -1;
    let mut cont_ind: i32 = -1;
    let mut mesh_ind: i32 = -1;
    let mut cap_skip_nz: Option<i32> = None;

    if version == IMOD_01 {
        imodel_read_v01(imod, file)?;
        imod_clean_surf(imod);
        return Ok(());
    }

    file.seek(SeekFrom::Start(0)).map_err(|_| IMOD_ERROR_READ)?;

    let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    if id != ID_IMOD {
        return Err(IMOD_ERROR_FORMAT);
    }

    let _ = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;

    let objsize = imodel_read_header(imod, file)?;

    if objsize != 0 {
        imod.obj = super::iobj::imod_objects_new(objsize).ok_or(IMOD_ERROR_MEMORY)?;
    } else {
        imod.obj = Vec::new();
    }

    obj_ind = -1;

    loop {
        let id = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
        match id {
            ID_OBJT => {
                obj_ind += 1;
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let (contsize, meshsize) =
                    imodel_read_object(&mut imod.obj[obj_ind as usize], file)?;
                if contsize != 0 {
                    imod.obj[obj_ind as usize].cont =
                        super::icont::imod_contours_new(contsize).ok_or(IMOD_ERROR_MEMORY)?;
                } else {
                    imod.obj[obj_ind as usize].cont = Vec::new();
                }
                cont_ind = -1;
                if meshsize != 0 {
                    if meshsize < 0 {
                        return Err(IMOD_ERROR_MEMORY);
                    }
                    // `imodMeshesNew` (`imesh.c:106`) zeroes every member.
                    imod.obj[obj_ind as usize].mesh = vec![Imesh::default(); meshsize as usize];
                }
                mesh_ind = -1;
            }

            ID_OLBL => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let mut error = 0;
                imod.obj[obj_ind as usize].label =
                    crate::imod::libimod::ilabel::imod_label_read(file, &mut error);
                if error != 0 {
                    return Err(error);
                }
            }

            ID_CONT => {
                cont_ind += 1;
                if obj_ind < 0
                    || obj_ind >= objsize
                    || cont_ind < 0
                    || cont_ind >= imod.obj[obj_ind as usize].cont.len() as i32
                {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                imodel_read_contour(
                    &mut imod.obj[obj_ind as usize].cont[cont_ind as usize],
                    file,
                )?;
            }

            ID_LABL => {
                if obj_ind < 0
                    || obj_ind >= objsize
                    || cont_ind < 0
                    || cont_ind >= imod.obj[obj_ind as usize].cont.len() as i32
                {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let mut error = 0;
                imod.obj[obj_ind as usize].cont[cont_ind as usize].label =
                    crate::imod::libimod::ilabel::imod_label_read(file, &mut error);
                if error != 0 {
                    return Err(error);
                }
            }

            ID_SIZE => {
                if obj_ind < 0
                    || obj_ind >= objsize
                    || cont_ind < 0
                    || cont_ind >= imod.obj[obj_ind as usize].cont.len() as i32
                {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                imodel_read_ptsizes(
                    &mut imod.obj[obj_ind as usize].cont[cont_ind as usize],
                    file,
                )?;
            }

            ID_MESH => {
                mesh_ind += 1;
                if obj_ind < 0
                    || obj_ind >= objsize
                    || mesh_ind < 0
                    || mesh_ind >= imod.obj[obj_ind as usize].mesh.len() as i32
                {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                imodel_read_mesh(
                    &mut imod.obj[obj_ind as usize].mesh[mesh_ind as usize],
                    file,
                )?;
            }

            ID_IEOF => break,

            ID_CLIP => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                imodel_read_clip(&mut imod.obj[obj_ind as usize], file, imod.flags);
            }

            /* DNM 9/4/02: pass flags so mat1 & mat3 can be read two ways */
            ID_IMAT => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let flags = imod.flags;
                imodel_read_imat(&mut imod.obj[obj_ind as usize], file, flags)?;
            }

            ID_VIEW => super::iview::imod_view_model_read(imod, file)?,

            ID_MCLP => super::iview::imod_view_clip_read(imod, file)?,

            /* image nat. transform */
            ID_IMNX => super::iview::imod_imnx_read(imod, file)?,

            /* Model general storage */
            ID_MOST => {
                let mut error = 0;
                imod.store = imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }

            /* Object general storage */
            ID_OBST => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let mut error = 0;
                imod.obj[obj_ind as usize].store =
                    imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }

            /* Contour general storage */
            ID_COST => {
                if obj_ind < 0
                    || obj_ind >= objsize
                    || cont_ind < 0
                    || cont_ind >= imod.obj[obj_ind as usize].cont.len() as i32
                {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let mut error = 0;
                imod.obj[obj_ind as usize].cont[cont_ind as usize].store =
                    imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }

            /* Mesh general storage */
            ID_MEST => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                if mesh_ind < 0 || mesh_ind >= imod.obj[obj_ind as usize].mesh.len() as i32 {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                let mut error = 0;
                imod.obj[obj_ind as usize].mesh[mesh_ind as usize].store =
                    imod_read_store(file, &mut error).ok_or(IMOD_ERROR_READ)?;
                if error != 0 {
                    return Err(error);
                }
            }

            ID_MEPA => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                cap_skip_nz = Some(imodel_read_meshparm(&mut imod.obj[obj_ind as usize], file)?);
            }

            ID_SKLI => {
                if obj_ind < 0 || obj_ind >= objsize {
                    return Err(IMOD_ERROR_CORRUPT);
                }
                imodel_read_meshskip(&mut imod.obj[obj_ind as usize], file, cap_skip_nz)?;
            }

            ID_SLAN => imodel_read_sliceang(imod, file)?,

            ID_OGRP => obj_group_read(&mut imod.group_list, file)?,

            _ => {
                let size = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)?;
                if size < 0 || file.seek(SeekFrom::Current(size as i64)).is_err() {
                    return Err(IMOD_ERROR_READ);
                }
            }
        }
    }

    if version < IMOD_V12 {
        imod_clean_surf(imod);
    }

    /* Make sure the current index is valid */
    if !imod.obj.is_empty() {
        imod.cindex.object = imod.cindex.object.max(0).min(imod.obj.len() as i32 - 1);
    } else {
        imod.cindex.object = -1;
    }
    if imod.cindex.object < 0 {
        imod.cindex.point = -1;
        imod.cindex.contour = -1;
    } else {
        imod.cindex.contour = imod
            .cindex
            .contour
            .min(imod.obj[imod.cindex.object as usize].cont.len() as i32 - 1)
            .max(-1);
    }
    if imod.cindex.contour < 0 {
        imod.cindex.point = -1;
    } else {
        imod.cindex.point = imod
            .cindex
            .point
            .min(
                imod.obj[imod.cindex.object as usize].cont[imod.cindex.contour as usize]
                    .pts
                    .len() as i32
                    - 1,
            )
            .max(-1);
    }

    Ok(())
}

/// Original: `imodReadFile` (`imodel_files.c:131`).
pub fn imod_read_file(imod: &mut Imod, file: &mut ImodFile) -> Result<(), i32> {
    file.seek(SeekFrom::Start(0)).map_err(|_| IMOD_ERROR_READ)?;
    if imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32 != ID_IMOD {
        return imod_read_ascii(imod, file);
    }
    let version = imod_get_int(file).map_err(|_| IMOD_ERROR_READ)? as u32;
    imodel_read(imod, file, version)
}

/// Original: `imodReadAscii` (`imodel_files.c:1181`).
///
/// The source reads with `imodFgetline`, which drops `\r`, skips blank lines
/// and skips comment lines beginning with `#`, so the ASCII a model writer
/// emits (which opens with a `#` banner and a blank line) parses only through
/// that routine.  Each keyword is matched with `substr` -- a bare prefix
/// compare, so `drawmode` matches both the model and the object line -- and
/// the value is taken with `sscanf`/`atoi`/`atof` at a fixed byte offset.
///
/// `sscanf` leaves an output untouched when its field fails to convert and
/// reports how many converted; the `parse` closures below do the same, and
/// `atoi`/`atof` return 0 for text they cannot read, as the C library does.
///
/// The translated `Imod` has no redundant `objsize` member; the object count
/// read from the header sizes `imod.obj` directly, and `slicerAngle` records
/// append to `imod.slicer_ang`.
pub fn imod_read_ascii(imod: &mut Imod, file: &mut ImodFile) -> Result<(), i32> {
    let mut line = [0u8; MAXLINE as usize];
    let mut ob: i32 = -1;
    let mut co: i32 = -1;
    let mut mh: i32 = -1;
    let mut valmin: f32 = 1.0e30;
    let mut valmax: f32 = -1.0e30;
    let mut objvmin: f32 = 0.;
    let mut objvmax: f32 = 0.;
    let mut got_obj_mm = 0;
    let mut view_ind: usize = 0;
    let mut obj_ind: usize = 0;

    // `atoi(&line[n])`: leading blanks, optional sign, digits; 0 if none.
    let atoi = |text: &str| -> i32 {
        let text = text.trim_start();
        let mut end = 0;
        for (i, ch) in text.char_indices() {
            if i == 0 && (ch == '+' || ch == '-') {
                end = i + ch.len_utf8();
                continue;
            }
            if ch.is_ascii_digit() {
                end = i + ch.len_utf8();
            } else {
                break;
            }
        }
        text[..end].parse::<i32>().unwrap_or(0)
    };
    // `atof(&line[n])`: the longest leading prefix that is a number; 0 if none.
    let atof = |text: &str| -> f64 {
        let text = text.trim_start();
        let mut best = 0.;
        let mut end = text.len();
        while end > 0 {
            if let Ok(value) = text[..end].parse::<f64>() {
                best = value;
                break;
            }
            end -= 1;
        }
        best
    };
    // `sscanf(line, "<keyword> %g %g ...", ...)`: split on whitespace after
    // the keyword and convert until one fails, reporting the values converted.
    let scan = |text: &str, skip: usize| -> Vec<f64> {
        let mut out = Vec::new();
        for word in text.split_whitespace().skip(skip) {
            match word.parse::<f64>() {
                Ok(value) => out.push(value),
                Err(_) => break,
            }
        }
        out
    };

    imod.cindex.object = -1;
    imod.cindex.contour = -1;
    imod.cindex.point = -1;

    file.seek(SeekFrom::Start(0)).map_err(|_| IMOD_ERROR_READ)?;
    let len = imod_fgetline(file, &mut line, MAXLINE);
    if len < 1 {
        return Err(-2);
    }
    if substr(&line, b"imod ") == 0 {
        return Err(-2);
    }

    let objsize = scan(&String::from_utf8_lossy(&line[..len as usize]), 1)
        .first()
        .copied()
        .unwrap_or(0.) as i32;
    imod.obj = imod_objects_new(objsize).unwrap_or_default();
    if objsize > 0 {
        imod.cindex.object = 0;
    }

    /* store.type = GEN_STORE_VALUE1; store.flags = GEN_STORE_FLOAT << 2 */
    let store_type: i16 = 10;
    let store_flags: u16 = 1 << 2;

    loop {
        let len = imod_fgetline(file, &mut line, MAXLINE);
        if len <= 0 {
            break;
        }
        let text = String::from_utf8_lossy(&line[..len as usize]).to_string();

        if substr(&line, b"contour ") != 0 {
            let vals = scan(&text, 1);
            co = vals.first().copied().unwrap_or(0.) as i32;
            let surf = vals.get(1).copied().unwrap_or(0.) as i32;
            let pts = vals.get(2).copied().unwrap_or(0.) as i32;
            if obj_ind >= imod.obj.len() || co < 0 || co >= imod.obj[obj_ind].cont.len() as i32 {
                return Err(-1);
            }
            {
                let cont = &mut imod.obj[obj_ind].cont[co as usize];
                cont.surf = surf;
                cont.pts = vec![Ipoint::default(); pts.max(0) as usize];
                cont.flags = 0;
                cont.time = 0;
            }
            if vals.len() > 3 {
                let value = vals[3] as f32;
                valmin = if valmin < value { valmin } else { value };
                valmax = if valmax > value { valmax } else { value };
                let mut store = Istore::default();
                store.type_ = store_type;
                store.flags = store_flags;
                store.index.set_i(co);
                store.value.set_f(value);
                if istore_insert(&mut imod.obj[obj_ind].store, store) != 0 {
                    return Err(-1);
                }
            }
            let mut active_val = 0;
            for pt in 0..pts.max(0) {
                let len = imod_fgetline(file, &mut line, MAXLINE);
                let text = String::from_utf8_lossy(&line[..len.max(0) as usize]).to_string();
                let vals = scan(&text, 0);
                let cont = &mut imod.obj[obj_ind].cont[co as usize];
                if let Some(value) = vals.first() {
                    cont.pts[pt as usize].x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    cont.pts[pt as usize].y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    cont.pts[pt as usize].z = *value as f32;
                }
                let size = vals.get(3).copied().unwrap_or(0.) as f32;
                if vals.len() > 3 && size >= 0. {
                    imod_point_set_size(&mut imod.obj[obj_ind].cont[co as usize], pt, size);
                }
                if vals.len() > 4 {
                    let value = vals[4] as f32;
                    let mut store = Istore::default();
                    store.type_ = store_type;
                    store.flags = store_flags;
                    store.index.set_i(pt);
                    store.value.set_f(value);
                    if istore_insert_change(&mut imod.obj[obj_ind].cont[co as usize].store, store)
                        != 0
                    {
                        return Err(-1);
                    }
                    active_val = 1;
                    valmin = if valmin < value { valmin } else { value };
                    valmax = if valmax > value { valmax } else { value };
                } else if active_val != 0 {
                    istore_end_change(
                        &mut imod.obj[obj_ind].cont[co as usize].store,
                        store_type,
                        pt,
                    );
                    active_val = 0;
                }
            }
            continue;
        }

        if substr(&line, b"mesh ") != 0 {
            let vals = scan(&text, 1);
            mh = vals.first().copied().unwrap_or(0.) as i32;
            let vsize = vals.get(1).copied().unwrap_or(0.) as i32;
            let lsize = vals.get(2).copied().unwrap_or(0.) as i32;
            if obj_ind >= imod.obj.len() || mh < 0 || mh >= imod.obj[obj_ind].mesh.len() as i32 {
                return Err(-1);
            }
            {
                let mesh = &mut imod.obj[obj_ind].mesh[mh as usize];
                mesh.flag = 0;
                mesh.time = 0;
                mesh.surf = 0;
                mesh.vert = vec![Ipoint::default(); vsize.max(0) as usize];
                mesh.list = vec![0; lsize.max(0) as usize];
            }
            for pt in 0..vsize.max(0) {
                let len = imod_fgetline(file, &mut line, MAXLINE);
                let text = String::from_utf8_lossy(&line[..len.max(0) as usize]).to_string();
                let vals = scan(&text, 0);
                let mesh = &mut imod.obj[obj_ind].mesh[mh as usize];
                if let Some(value) = vals.first() {
                    mesh.vert[pt as usize].x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    mesh.vert[pt as usize].y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    mesh.vert[pt as usize].z = *value as f32;
                }
            }
            for i in 0..lsize.max(0) {
                let len = imod_fgetline(file, &mut line, MAXLINE);
                let text = String::from_utf8_lossy(&line[..len.max(0) as usize]).to_string();
                if let Some(value) = scan(&text, 0).first() {
                    imod.obj[obj_ind].mesh[mh as usize].list[i as usize] = *value as i32;
                }
            }
            continue;
        }

        if substr(&line, b"object ") != 0 {
            /* new object: put out min/max for last one if any */
            if ob >= 0 && (got_obj_mm != 0 || valmin <= valmax) && obj_ind < imod.obj.len() {
                istore_add_min_max(
                    &mut imod.obj[obj_ind].store,
                    11,
                    if got_obj_mm != 0 { objvmin } else { valmin },
                    if got_obj_mm != 0 { objvmax } else { valmax },
                );
            }
            let vals = scan(&text, 1);
            ob = vals.first().copied().unwrap_or(0.) as i32;
            let conts = vals.get(1).copied().unwrap_or(0.) as i32;
            let meshes = vals.get(2).copied().unwrap_or(0.) as i32;

            if ob < 0 {
                return Err(-1);
            }
            if ob > imod.obj.len() as i32 {
                return Err(-1);
            }
            // `imodel_files.c:1307` tests `ob > imod->objsize`, so `ob` equal to
            // the count indexes one past the array in the source.
            if ob >= imod.obj.len() as i32 {
                return Err(-1);
            }
            obj_ind = ob as usize;
            if conts != 0 {
                imod.obj[obj_ind].cont = imod_contours_new(conts).unwrap_or_default();
            }
            if meshes != 0 {
                imod.obj[obj_ind].mesh = imod_meshes_new(meshes).unwrap_or_default();
            }
            mh = -1;
            co = -1;
            valmin = 1.0e30;
            valmax = -1.0e30;
            got_obj_mm = 0;
            continue;
        }

        if substr(&line, b"color ") != 0 {
            let vals = scan(&text, 1);
            if obj_ind < imod.obj.len() {
                let obj = &mut imod.obj[obj_ind];
                if let Some(value) = vals.first() {
                    obj.red = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    obj.green = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    obj.blue = *value as f32;
                }
                if let Some(value) = vals.get(3) {
                    obj.trans = *value as i32 as u8;
                }
            }
        }
        if substr(&line, b"Fillcolor ") != 0 {
            let vals = scan(&text, 1);
            if obj_ind < imod.obj.len() {
                let obj = &mut imod.obj[obj_ind];
                if let Some(value) = vals.first() {
                    obj.fillred = *value as i32 as u8;
                }
                if let Some(value) = vals.get(1) {
                    obj.fillgreen = *value as i32 as u8;
                }
                if let Some(value) = vals.get(2) {
                    obj.fillblue = *value as i32 as u8;
                }
            }
        }

        if obj_ind < imod.obj.len() {
            let obj = &mut imod.obj[obj_ind];
            if substr(&line, b"open") != 0 {
                obj.flags |= IMOD_OBJFLAG_OPEN;
            }
            if substr(&line, b"closed") != 0 {
                obj.flags &= !IMOD_OBJFLAG_OPEN;
            }
            if substr(&line, b"fill") != 0 {
                obj.flags |= IMOD_OBJFLAG_FILL;
            }
            if substr(&line, b"scattered") != 0 {
                obj.flags |= IMOD_OBJFLAG_SCAT;
            }
            if substr(&line, b"insideout") != 0 {
                obj.flags |= IMOD_OBJFLAG_OUT;
            }
            if substr(&line, b"drawmesh") != 0 {
                obj.flags |= IMOD_OBJFLAG_MESH;
            }
            if substr(&line, b"nolines") != 0 {
                obj.flags |= IMOD_OBJFLAG_NOLINE;
            }
            if substr(&line, b"bothsides") != 0 {
                obj.flags |= IMOD_OBJFLAG_TWO_SIDE;
            }
            if substr(&line, b"usefill") != 0 {
                obj.flags |= IMOD_OBJFLAG_FCOLOR;
            }
            if substr(&line, b"pntusefill") != 0 {
                obj.flags |= IMOD_OBJFLAG_FCOLOR_PNT;
            }
            if substr(&line, b"pntonsec") != 0 {
                obj.flags |= IMOD_OBJFLAG_PNT_ON_SEC;
            }
            if substr(&line, b"antialias") != 0 {
                obj.flags |= IMOD_OBJFLAG_ANTI_ALIAS;
            }
            if substr(&line, b"hastimes") != 0 {
                obj.flags |= IMOD_OBJFLAG_TIME;
            }
            if substr(&line, b"usevalue") != 0 {
                obj.flags |= IMOD_OBJFLAG_USE_VALUE;
            }
            if substr(&line, b"valcolor") != 0 {
                obj.flags |= IMOD_OBJFLAG_MCOLOR;
            }
        }

        if substr(&line, b"offsets ") != 0 {
            let vals = scan(&text, 1);
            if let Some(value) = vals.first() {
                imod.xoffset = *value as f32;
            }
            if let Some(value) = vals.get(1) {
                imod.yoffset = *value as f32;
            }
            if let Some(value) = vals.get(2) {
                imod.zoffset = *value as f32;
            }
        }

        if substr(&line, b"angles ") != 0 {
            let vals = scan(&text, 1);
            if let Some(value) = vals.first() {
                imod.alpha = *value as f32;
            }
            if let Some(value) = vals.get(1) {
                imod.beta = *value as f32;
            }
            if let Some(value) = vals.get(2) {
                imod.gamma = *value as f32;
            }
        }

        if substr(&line, b"refcurscale ") != 0 {
            if imod.ref_image.is_none() {
                imod.ref_image = imod_imnx_new();
            }
            let vals = scan(&text, 1);
            if let Some(reference) = imod.ref_image.as_mut() {
                if let Some(value) = vals.first() {
                    reference.cscale.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    reference.cscale.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    reference.cscale.z = *value as f32;
                }
            }
        }
        if substr(&line, b"refcurtrans ") != 0 {
            if imod.ref_image.is_none() {
                imod.ref_image = imod_imnx_new();
            }
            let vals = scan(&text, 1);
            if let Some(reference) = imod.ref_image.as_mut() {
                if let Some(value) = vals.first() {
                    reference.ctrans.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    reference.ctrans.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    reference.ctrans.z = *value as f32;
                }
            }
        }
        if substr(&line, b"refcurrot ") != 0 {
            if imod.ref_image.is_none() {
                imod.ref_image = imod_imnx_new();
            }
            imod.flags |= IMODF_TILTOK;
            let vals = scan(&text, 1);
            if let Some(reference) = imod.ref_image.as_mut() {
                if let Some(value) = vals.first() {
                    reference.crot.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    reference.crot.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    reference.crot.z = *value as f32;
                }
            }
        }
        if substr(&line, b"refoldtrans ") != 0 {
            if imod.ref_image.is_none() {
                imod.ref_image = imod_imnx_new();
            }
            imod.flags |= IMODF_OTRANS_ORIGIN;
            let vals = scan(&text, 1);
            if let Some(reference) = imod.ref_image.as_mut() {
                if let Some(value) = vals.first() {
                    reference.otrans.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    reference.otrans.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    reference.otrans.z = *value as f32;
                }
            }
        }

        if substr(&line, b"nodraw") != 0 && obj_ind < imod.obj.len() {
            imod.obj[obj_ind].flags |= IMOD_OBJFLAG_OFF;
        }

        if substr(&line, b"scale ") != 0 {
            let vals = scan(&text, 1);
            if let Some(value) = vals.first() {
                imod.xscale = *value as f32;
            }
            if let Some(value) = vals.get(1) {
                imod.yscale = *value as f32;
            }
            if let Some(value) = vals.get(2) {
                imod.zscale = *value as f32;
            }
        }

        if substr(&line, b"max ") != 0 {
            let vals = scan(&text, 1);
            if let Some(value) = vals.first() {
                imod.xmax = *value as i32;
            }
            if let Some(value) = vals.get(1) {
                imod.ymax = *value as i32;
            }
            if let Some(value) = vals.get(2) {
                imod.zmax = *value as i32;
            }
        }

        if substr(&line, b"mousemode") != 0 {
            imod.mousemode = atoi(&text[9.min(text.len())..]);
        }

        if substr(&line, b"drawmode") != 0 {
            imod.drawmode = atoi(&text[8.min(text.len())..]);
        }

        /* substr(line, "") is vacuously true, so every line is scanned for
        the b&w levels; only a line that starts with them converts. */
        {
            let vals = text
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|word| !word.is_empty())
                .skip(1)
                .take_while(|word| word.parse::<f64>().is_ok())
                .map(|word| word.parse::<f64>().unwrap())
                .collect::<Vec<_>>();
            if substr(&line, b"b&w_level ") != 0 {
                if let Some(value) = vals.first() {
                    imod.blacklevel = *value as i32;
                }
                if let Some(value) = vals.get(1) {
                    imod.whitelevel = *value as i32;
                }
            }
        }

        if substr(&line, b"resolution") != 0 {
            imod.res = atof(&text[10.min(text.len())..]) as f32 as i32;
        }

        if substr(&line, b"threshold") != 0 {
            imod.thresh = atoi(&text[9.min(text.len())..]);
        }

        if substr(&line, b"pixsize") != 0 {
            imod.pixsize = atof(&text[7.min(text.len())..]) as f32;
        }

        if substr(&line, b"flipped") != 0 && atoi(&text[7.min(text.len())..]) != 0 {
            imod.flags |= IMODF_FLIPYZ;
        }

        if substr(&line, b"units") != 0 {
            if text.contains("mm") {
                imod.units = IMOD_UNIT_MM;
            }
            if text.contains("um") {
                imod.units = IMOD_UNIT_UM;
            }
            if text.contains("nm") {
                imod.units = IMOD_UNIT_NM;
            }
        }

        if substr(&line, b"currentview") != 0 {
            imod.cview = atoi(&text[11.min(text.len())..]);
        }

        if substr(&line, b"slicerAngle") != 0 {
            let mut slan = Slicer_angles::default();
            let vals = scan(&text, 1);
            if let Some(value) = vals.first() {
                slan.time = *value as i32;
            }
            for i in 0..3 {
                if let Some(value) = vals.get(1 + i) {
                    slan.angles[i] = *value as f32;
                }
            }
            if let Some(value) = vals.get(4) {
                slan.center.x = *value as f32;
            }
            if let Some(value) = vals.get(5) {
                slan.center.y = *value as f32;
            }
            if let Some(value) = vals.get(6) {
                slan.center.z = *value as f32;
            }
            /* Skip eight blanks, then copy at most ANGLE_STRSIZE bytes. */
            let bytes = text.as_bytes();
            let mut strptr: Option<usize> = Some(0);
            for _ in 0..8 {
                strptr = match strptr {
                    Some(at) if at + 1 <= bytes.len() => bytes[at + 1..]
                        .iter()
                        .position(|&b| b == b' ')
                        .map(|off| at + 1 + off),
                    _ => None,
                };
                if strptr.is_none() {
                    break;
                }
            }
            if let Some(at) = strptr {
                let src = &bytes[(at + 1).min(bytes.len())..];
                let take = src.len().min(ANGLE_STRSIZE);
                slan.label[..take].copy_from_slice(&src[..take]);
            }
            slan.label[ANGLE_STRSIZE - 1] = 0;
            imod.slicer_ang.push(slan);
        }

        if substr(&line, b"name") != 0 && obj_ind < imod.obj.len() {
            let bytes = text.as_bytes();
            let obj = &mut imod.obj[obj_ind];
            let mut i = 0;
            while i + 5 < bytes.len()
                && i < IOBJ_STRSIZE
                && bytes[i + 5] != b'\r'
                && bytes[i + 5] != b'\n'
            {
                obj.name[i] = bytes[i + 5];
                i += 1;
            }
            if i < IOBJ_STRSIZE {
                obj.name[i] = 0;
            }
        }

        if obj_ind < imod.obj.len() {
            let obj = &mut imod.obj[obj_ind];
            if substr(&line, b"linewidth") != 0 {
                obj.linewidth = atoi(&text[9.min(text.len())..]) as u8;
            }
            if substr(&line, b"surfsize") != 0 {
                obj.surfsize = atoi(&text[8.min(text.len())..]);
            }
            if substr(&line, b"pointsize") != 0 {
                obj.pdrawsize = atoi(&text[9.min(text.len())..]);
            }
            if substr(&line, b"axis") != 0 {
                obj.axis = atoi(&text[4.min(text.len())..]);
            }
            if substr(&line, b"drawmode") != 0 {
                obj.drawmode = atoi(&text[8.min(text.len())..]);
            }
            if substr(&line, b"width2D") != 0 {
                obj.linewidth2 = atoi(&text[7.min(text.len())..]) as u8;
            }
            if substr(&line, b"symbol") != 0 {
                obj.symbol = atoi(&text[6.min(text.len())..]) as u8;
            }
            if substr(&line, b"symsize") != 0 {
                obj.symsize = atoi(&text[7.min(text.len())..]) as u8;
            }
            if substr(&line, b"symflags") != 0 {
                obj.symflags = atoi(&text[8.min(text.len())..]) as u8;
            }
            if substr(&line, b"ambient") != 0 {
                obj.ambient = atoi(&text[7.min(text.len())..]) as u8;
            }
            if substr(&line, b"diffuse") != 0 {
                obj.diffuse = atoi(&text[7.min(text.len())..]) as u8;
            }
            if substr(&line, b"specular") != 0 {
                obj.specular = atoi(&text[8.min(text.len())..]) as u8;
            }
            if substr(&line, b"shininess") != 0 {
                obj.shininess = atoi(&text[9.min(text.len())..]) as u8;
            }
            if substr(&line, b"obquality") != 0 {
                obj.quality = atoi(&text[9.min(text.len())..]) as u8;
            }
            if substr(&line, b"valblack") != 0 {
                obj.valblack = atoi(&text[8.min(text.len())..]) as u8;
            }
            if substr(&line, b"valwhite") != 0 {
                obj.valwhite = atoi(&text[8.min(text.len())..]) as u8;
            }
            if substr(&line, b"meshthick") != 0 {
                obj.mesh_thickness = atoi(&text[9.min(text.len())..]) as u8;
            }
            if substr(&line, b"matflags2") != 0 {
                obj.matflags2 = atoi(&text[9.min(text.len())..]) as u8;
            }
        }

        if substr(&line, b"valminmax") != 0 {
            let vals = scan(&text, 1);
            if let Some(value) = vals.first() {
                objvmin = *value as f32;
            }
            if let Some(value) = vals.get(1) {
                objvmax = *value as f32;
            }
            got_obj_mm = 1;
        }

        if substr(&line, b"objclips") != 0 && obj_ind < imod.obj.len() {
            let mut clips = imod.obj[obj_ind].clips.clone();
            read_ascii_clips(file, &mut line, &mut clips, "objclips");
            imod.obj[obj_ind].clips = clips;
        }

        if substr(&line, b"contflags") != 0
            && co >= 0
            && obj_ind < imod.obj.len()
            && co < imod.obj[obj_ind].cont.len() as i32
        {
            imod.obj[obj_ind].cont[co as usize].flags =
                atof(&text[9.min(text.len())..]) as f32 as u32;
        }

        if substr(&line, b"conttime") != 0
            && co >= 0
            && obj_ind < imod.obj.len()
            && co < imod.obj[obj_ind].cont.len() as i32
        {
            imod.obj[obj_ind].cont[co as usize].time = atoi(&text[8.min(text.len())..]);
        }

        if substr(&line, b"Meshflags") != 0
            && mh >= 0
            && obj_ind < imod.obj.len()
            && mh < imod.obj[obj_ind].mesh.len() as i32
        {
            imod.obj[obj_ind].mesh[mh as usize].flag =
                atof(&text[9.min(text.len())..]) as f32 as u32;
        }

        if substr(&line, b"Meshtime") != 0
            && mh >= 0
            && obj_ind < imod.obj.len()
            && mh < imod.obj[obj_ind].mesh.len() as i32
        {
            imod.obj[obj_ind].mesh[mh as usize].time = atoi(&text[8.min(text.len())..]) as i16;
        }

        if substr(&line, b"Meshsurf") != 0
            && mh >= 0
            && obj_ind < imod.obj.len()
            && mh < imod.obj[obj_ind].mesh.len() as i32
        {
            imod.obj[obj_ind].mesh[mh as usize].surf = atoi(&text[8.min(text.len())..]) as i16;
        }

        if substr(&line, b"view ") != 0 {
            if imod_view_model_new(imod) != 0 {
                return Err(-1);
            }
            view_ind = imod.view.len() - 1;
        }

        if substr(&line, b"globalclips") != 0 && view_ind < imod.view.len() {
            let mut clips = imod.view[view_ind].clips.clone();
            read_ascii_clips(file, &mut line, &mut clips, "globalclips");
            imod.view[view_ind].clips = clips;
        }

        if view_ind < imod.view.len() {
            if substr(&line, b"viewfovy") != 0 {
                imod.view[view_ind].fovy = atof(&text[8.min(text.len())..]) as f32;
            }
            if substr(&line, b"viewcnear") != 0 {
                imod.view[view_ind].cnear = atof(&text[9.min(text.len())..]) as f32;
            }
            if substr(&line, b"viewcfar") != 0 {
                imod.view[view_ind].cfar = atof(&text[8.min(text.len())..]) as f32;
            }
            if substr(&line, b"viewflags") != 0 {
                imod.view[view_ind].world = atof(&text[9.min(text.len())..]) as f32 as u32;
            }
            if substr(&line, b"viewscale ") != 0 {
                let vals = scan(&text, 1);
                if let Some(value) = vals.first() {
                    imod.view[view_ind].scale.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    imod.view[view_ind].scale.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    imod.view[view_ind].scale.z = *value as f32;
                }
            }
            if substr(&line, b"viewtrans ") != 0 {
                let vals = scan(&text, 1);
                if let Some(value) = vals.first() {
                    imod.view[view_ind].trans.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    imod.view[view_ind].trans.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    imod.view[view_ind].trans.z = *value as f32;
                }
            }
            if substr(&line, b"viewrot ") != 0 {
                let vals = scan(&text, 1);
                if let Some(value) = vals.first() {
                    imod.view[view_ind].rot.x = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    imod.view[view_ind].rot.y = *value as f32;
                }
                if let Some(value) = vals.get(2) {
                    imod.view[view_ind].rot.z = *value as f32;
                }
            }
            if substr(&line, b"viewlight ") != 0 {
                let vals = scan(&text, 1);
                if let Some(value) = vals.first() {
                    imod.view[view_ind].lightx = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    imod.view[view_ind].lighty = *value as f32;
                }
            }
            if substr(&line, b"depthcue ") != 0 {
                let vals = scan(&text, 1);
                if let Some(value) = vals.first() {
                    imod.view[view_ind].dcstart = *value as f32;
                }
                if let Some(value) = vals.get(1) {
                    imod.view[view_ind].dcend = *value as f32;
                }
            }
            if substr(&line, b"viewlabel") != 0 {
                let bytes = text.as_bytes();
                let view = &mut imod.view[view_ind];
                let mut i = 0;
                while i + 10 < bytes.len()
                    && i < VIEW_STRSIZE
                    && bytes[i + 10] != b'\r'
                    && bytes[i + 10] != b'\n'
                {
                    view.label[i] = bytes[i + 10];
                    i += 1;
                }
                if i < VIEW_STRSIZE {
                    view.label[i] = 0;
                }
            }
        }
    }

    if ob >= 0 && (got_obj_mm != 0 || valmin <= valmax) && obj_ind < imod.obj.len() {
        istore_add_min_max(
            &mut imod.obj[obj_ind].store,
            11,
            if got_obj_mm != 0 { objvmin } else { valmin },
            if got_obj_mm != 0 { objvmax } else { valmax },
        );
    }
    imod.cview = imod.cview.clamp(0, imod.view.len() as i32 - 1);
    Ok(())
}

/// Original: `imodRead` (`imodel_files.c:171`).
pub fn imod_read(path: impl AsRef<Path>) -> Result<Imod, i32> {
    // `imodel_files.c:173` builds the model with `imodNew`, so the defaults
    // `imodDefault` sets -- the "IMOD-NewModel" name, the initial view, the
    // pixel size -- are in place before the file is read over them.  A binary
    // model overwrites all of them; an ASCII one leaves whatever it omits.
    let mut imod = match crate::imod::libimod::imodel::imod_new() {
        Some(imod) => imod,
        None => return Err(IMOD_ERROR_MEMORY),
    };
    let mut file =
        ImodFile::open(path.as_ref().to_str().unwrap_or(""), "rb").ok_or(IMOD_ERROR_READ)?;
    imod_read_file(&mut imod, &mut file)?;
    imod.file_name = Some(path.as_ref().to_string_lossy().into_owned());
    Ok(imod)
}

/// Original: `imodel_write_contour` (`imodel_files.c:449`).
pub fn imodel_write_contour(cont: &Icont, file: &mut ImodFile) -> Result<(), i32> {
    imod_put_int(file, ID_CONT as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.pts.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.flags as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.time).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, cont.surf).map_err(|_| IMOD_ERROR_WRITE)?;
    for point in &cont.pts {
        imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    if cont.label.is_some() {
        crate::imod::libimod::ilabel::imod_label_write(cont.label.as_ref(), ID_LABL, file);
    }
    if !cont.sizes.is_empty() {
        imod_put_int(file, ID_SIZE as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_int(file, (cont.sizes.len() * 4) as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        for size in &cont.sizes {
            imod_put_float(file, *size).map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    if imod_write_store(&cont.store, ID_COST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    Ok(())
}

/// Original: `imodel_write_mesh` (`imodel_files.c:480`).
pub fn imodel_write_mesh(mesh: &Imesh, file: &mut ImodFile) -> Result<(), i32> {
    // imeshThickness (`imesh.h:54`): paired thickness meshes are not written.
    if (mesh.flag & (63 << 24)) >> 24 != 0 {
        return Ok(());
    }
    imod_put_int(file, ID_MESH as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, mesh.vert.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, mesh.list.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, mesh.flag as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_short(file, mesh.time).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_short(file, mesh.surf).map_err(|_| IMOD_ERROR_WRITE)?;
    for point in &mesh.vert {
        imod_put_float(file, point.x).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.y).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_float(file, point.z).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for list in &mesh.list {
        imod_put_int(file, *list).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    if imod_write_store(&mesh.store, ID_MEST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    Ok(())
}

/// Original: `imodel_write_object` (`imodel_files.c:345`).
pub fn imodel_write_object(
    object: &Iobj,
    file: &mut ImodFile,
    skip_mesh: i32,
    scale: &Ipoint,
) -> Result<(), i32> {
    // `imodel_files.c:352` writes the number of meshes that have no thickness
    // in their flag word, not the size of the mesh array.
    let mut num_real = 0;
    for mesh in &object.mesh {
        // imeshThickness (`imesh.h:54`)
        if (mesh.flag & (63 << 24)) >> 24 == 0 {
            num_real += 1;
        }
    }
    imod_put_int(file, ID_OBJT as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    // `imodel_files.c:352` writes the whole IOBJ_STRSIZE array, including any
    // bytes past the terminating NUL.
    file.write_all(&object.name).map_err(|_| IMOD_ERROR_WRITE)?;
    for value in object.extra {
        imod_put_int(file, value as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_int(file, object.cont.len() as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.flags as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.axis).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.drawmode).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_float(file, object.red).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_float(file, object.green).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_float(file, object.blue).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.pdrawsize).map_err(|_| IMOD_ERROR_WRITE)?;
    for value in [
        object.symbol,
        object.symsize,
        object.linewidth2,
        object.linewidth,
        object.linesty,
        object.symflags,
        object.sympad,
    ] {
        imod_put_byte(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_byte(file, object.trans).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, num_real).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.surfsize).map_err(|_| IMOD_ERROR_WRITE)?;
    if object.label.is_some() {
        crate::imod::libimod::ilabel::imod_label_write(object.label.as_ref(), ID_OLBL, file);
    }
    for contour in &object.cont {
        imodel_write_contour(contour, file)?;
    }
    if (!object.cont.is_empty() && (skip_mesh & 1) == 0)
        || (object.cont.is_empty() && (skip_mesh & 2) == 0)
    {
        for mesh in &object.mesh {
            imodel_write_mesh(mesh, file)?;
        }
    }
    if object.clips.count > 0 {
        imod_put_int(file, ID_CLIP as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_int(file, 4 + object.clips.count as i32 * 24).map_err(|_| IMOD_ERROR_WRITE)?;
        file.write_all(&[
            object.clips.count,
            object.clips.flags,
            object.clips.trans,
            object.clips.plane,
        ])
        .map_err(|_| IMOD_ERROR_WRITE)?;
        // `imodel_files.c:406-408` makes two `imodPutScaledPoints` calls, so
        // all `count` normals precede all `count` points; they are not
        // interleaved. Normals use the inverse of the model binning scale.
        let count = (object.clips.count as usize).min(object.clips.normal.len());
        let norm_scale = Ipoint {
            x: 1. / scale.x,
            y: 1. / scale.y,
            z: 1. / scale.z,
        };
        imod_put_scaled_points(file, &object.clips.normal, count as i32, &norm_scale)
            .map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_scaled_points(file, &object.clips.point, count as i32, scale)
            .map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_int(file, ID_IMAT as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, 16).map_err(|_| IMOD_ERROR_WRITE)?;
    file.write_all(&[
        object.ambient,
        object.diffuse,
        object.specular,
        object.shininess,
    ])
    .map_err(|_| IMOD_ERROR_WRITE)?;
    file.write_all(&[
        object.fillred,
        object.fillgreen,
        object.fillblue,
        object.quality,
    ])
    .map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, object.mat2 as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    file.write_all(&[
        object.valblack,
        object.valwhite,
        object.matflags2,
        object.mesh_thickness,
    ])
    .map_err(|_| IMOD_ERROR_WRITE)?;
    if let Some(params) = object.mesh_param.as_ref() {
        // `imodel_files.c:423-438`: MEPA is the contiguous nine-int,
        // ten-float `MeshParams` prefix, followed by SKLI when caps skip Z.
        imod_put_int(file, ID_MEPA as i32).map_err(|_| IMOD_ERROR_WRITE)?;
        imod_put_int(file, 76).map_err(|_| IMOD_ERROR_WRITE)?;
        let ints = [
            params.flags as i32,
            params.cap,
            params.passes,
            params.cap_skip_nz,
            params.incz_low_res,
            params.incz_high_res,
            params.minz,
            params.maxz,
            params.spare_int,
        ];
        imod_put_ints(file, &ints, 9).map_err(|_| IMOD_ERROR_WRITE)?;
        let floats = [
            params.overlap,
            params.tube_diameter,
            params.xmin,
            params.xmax,
            params.ymin,
            params.ymax,
            params.tol_low_res,
            params.tol_high_res,
            params.flat_crit,
            params.spare_float,
        ];
        imod_put_floats(file, &floats, 10).map_err(|_| IMOD_ERROR_WRITE)?;
        if params.cap_skip_nz != 0 {
            let Some(list) = params.cap_skip_zlist.as_ref() else {
                return Err(IMOD_ERROR_WRITE);
            };
            if list.len() != params.cap_skip_nz as usize {
                return Err(IMOD_ERROR_WRITE);
            }
            imod_put_int(file, ID_SKLI as i32).map_err(|_| IMOD_ERROR_WRITE)?;
            imod_put_int(file, params.cap_skip_nz * 4).map_err(|_| IMOD_ERROR_WRITE)?;
            imod_put_ints(file, list, params.cap_skip_nz).map_err(|_| IMOD_ERROR_WRITE)?;
        }
    }
    if imod_write_store(&object.store, ID_OBST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    Ok(())
}

/// Original: `imodel_write` (`imodel_files.c:276`).
pub fn imodel_write(imod: &Imod, file: &mut ImodFile, skip_mesh: i32) -> Result<(), i32> {
    file.seek(SeekFrom::Start(0))
        .map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, ID_IMOD as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, IMOD_V12 as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    // `imodel_files.c:300` writes the whole IMOD_STRSIZE array, including any
    // bytes past the terminating NUL.
    file.write_all(&imod.name).map_err(|_| IMOD_ERROR_WRITE)?;
    // `imodel_files.c:286` sets these bits before serializing the header.
    // They declare the byte material fields, multiple clip-plane support, and
    // mesh-thickness field emitted by this writer.
    let flags = imod.flags | (1 << 13) | (1 << 12) | (1 << 9);
    for value in [
        imod.xmax,
        imod.ymax,
        imod.zmax,
        imod.obj.len() as i32,
        flags as i32,
        imod.drawmode,
        imod.mousemode,
        imod.blacklevel,
        imod.whitelevel,
    ] {
        imod_put_int(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for value in [
        imod.xoffset,
        imod.yoffset,
        imod.zoffset,
        imod.xscale,
        imod.yscale,
        imod.zscale,
    ] {
        imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    for value in [
        imod.cindex.object,
        imod.cindex.contour,
        imod.cindex.point,
        imod.res,
        imod.thresh,
    ] {
        imod_put_int(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    imod_put_float(file, imod.pixsize).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, imod.units).map_err(|_| IMOD_ERROR_WRITE)?;
    imod_put_int(file, imod.csum).map_err(|_| IMOD_ERROR_WRITE)?;
    for value in [imod.alpha, imod.beta, imod.gamma] {
        imod_put_float(file, value).map_err(|_| IMOD_ERROR_WRITE)?;
    }
    let scale = Ipoint {
        x: imod.xybin as f32,
        y: imod.xybin as f32,
        z: imod.zbin as f32,
    };
    for object in &imod.obj {
        imodel_write_object(object, file, skip_mesh, &scale)?;
    }
    super::iview::imod_view_model_write(imod, file)?;
    super::iview::imod_imnx_write(imod, file)?;
    obj_group_list_write(&imod.group_list, file)?;
    if imod_write_store(&imod.store, ID_MOST as i32, file) != 0 {
        return Err(IMOD_ERROR_WRITE);
    }
    imod_put_int(file, ID_IEOF as i32).map_err(|_| IMOD_ERROR_WRITE)?;
    file.flush().map_err(|_| IMOD_ERROR_WRITE)
}

/// Original: `imodWrite` (`imodel_files.c:249`).
pub fn imod_write(imod: &Imod, file: &mut ImodFile) -> Result<(), i32> {
    imod_write_skip_mesh(imod, file, 0)
}

/// Original: `imodWriteSkipMesh` (`imodel_files.c:259`).
///
/// The source saves and restores `imod->file` around the write; Rust callers
/// own that borrow explicitly, so there is no aliased file member to restore.
pub fn imod_write_skip_mesh(imod: &Imod, fout: &mut ImodFile, which_skip: i32) -> Result<(), i32> {
    if imodel_write(imod, fout, which_skip).is_err() {
        return Err(-1);
    }
    Ok(())
}

/// Original: `imodWriteFile` (`imodel_files.c:236`).
///
/// The source writes to the `FILE *` stored in `imod->file`; Rust passes the
/// destination explicitly instead of storing an aliased handle in the model.
pub fn imod_write_file(imod: &Imod, file: &mut ImodFile) -> Result<(), i32> {
    imodel_write(imod, file, 0)
}

/// Original: `imodFileWrite` (`imodel_files.c:77`).
pub fn imod_file_write(imod: &Imod, path: impl AsRef<Path>) -> Result<(), i32> {
    let mut file =
        ImodFile::open(path.as_ref().to_str().unwrap_or(""), "wb").ok_or(IMOD_ERROR_WRITE)?;
    imod_write(imod, &mut file)
}

/// Original: `imodFileRead` (`imodel_files.c:68`).
pub fn imod_file_read(filename: impl AsRef<Path>) -> Result<Imod, i32> {
    imod_read(filename)
}

/// Original: `imodOpenFile` (`imodel_files.c:93`).
///
/// The source stores the opened `FILE *` in `imod->file` and a copy of the path
/// in `imod->fileName`; Rust returns the handle to its caller, while retaining
/// the source filename in `imod.file_name`. `mode` is the `fopen` mode string: the first character selects read,
/// write or append and a `+` adds the other direction.
pub fn imod_open_file(filename: &str, mode: &str, imod: &mut Imod) -> Result<ImodFile, i32> {
    let file = ImodFile::open(filename, mode).ok_or(-1)?;
    imod.file_name = Some(filename.to_owned());
    Ok(file)
}

/// Original: `imodCloseFile` (`imodel_files.c:115`).
///
/// The source closes `imod->file`; that member has no counterpart here, so the
/// handle opened by `imod_open_file` is passed back in.
pub fn imod_close_file(file: Option<ImodFile>) -> i32 {
    let Some(file) = file else {
        return -1;
    };
    drop(file);
    0
}

/// Original: `imodTestIfModelFile` (`imodel_files.c:212`).
///
/// `substr` (`imodel_from.c:207`) compares the first `strlen(ls)` characters of
/// the two strings; `imodel_from.c` has no translated module, so the comparison
/// is done in place.
pub fn imod_test_if_model_file(filename: impl AsRef<Path>) -> i32 {
    let mut ret = 0;
    let mut line = [0u8; MAXLINE as usize];
    let Some(mut fp) = ImodFile::open(filename.as_ref().to_str().unwrap_or(""), "r") else {
        return 1;
    };
    let id = imod_get_int(&mut fp).unwrap_or(0) as u32;
    if id != ID_IMOD {
        if fp.seek(SeekFrom::Start(0)).is_err() {
            return 1;
        }
        let len = imod_fgetline(&mut fp, &mut line, MAXLINE);
        /* substr(line, "imod ") */
        let prefix = b"imod ";
        let mut matched = 1;
        for i in 0..prefix.len() {
            if line[i] != prefix[i] {
                matched = 0;
                break;
            }
        }
        if len < 1 || matched == 0 {
            ret = 2;
        } else {
            ret = -1;
        }
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only: builds a fixed-size NUL-padded model/object name array.
    fn name_array<const N: usize>(text: &str) -> [u8; N] {
        let mut name = [0; N];
        for (slot, byte) in name.iter_mut().zip(text.as_bytes()) {
            *slot = *byte;
        }
        name
    }

    #[test]
    fn object_group_ogrp_chunk_round_trips_in_a_real_binary_model() {
        let path = std::env::temp_dir().join(format!("imod-rs-ogrp-{}.mod", std::process::id()));
        let model = Imod {
            group_list: vec![super::super::imodel::Iobj_group {
                name: *b"selected objects\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
                obj_list: vec![0, 2, 5],
            }],
            ..Imod::default()
        };
        imod_file_write(&model, &path).unwrap();
        let decoded = imod_read(&path).unwrap();
        assert_eq!(decoded.group_list, model.group_list);
        assert_eq!(decoded.file_name.as_deref(), path.to_str());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn binary_clip_chunk_uses_source_model_binning_scales() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-binned-clip-{}.mod",
            std::process::id()
        ));
        let mut model = Imod {
            xybin: 2,
            zbin: 3,
            obj: vec![Iobj::default()],
            ..Imod::default()
        };
        let clips = &mut model.obj[0].clips;
        clips.count = 1;
        clips.normal[0] = Ipoint { x: 2., y: 4., z: 6. };
        clips.point[0] = Ipoint { x: 1., y: 2., z: 3. };
        imod_file_write(&model, &path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let clip = bytes.windows(4).position(|window| window == b"CLIP").unwrap();
        // chunk ID + size + four clip bytes, then all normals followed by points
        let floats = &bytes[clip + 12..clip + 36];
        let decoded = floats
            .chunks_exact(4)
            .map(|value| f32::from_be_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(decoded, vec![1., 2., 2., 2., 4., 9.]);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writer_sets_source_format_flags_and_skips_default_view_chunk() {
        let path = std::env::temp_dir().join(format!(
            "imod-rs-imodel-write-default-view-{}",
            std::process::id()
        ));
        let mut file = ImodFile::open(path.to_str().unwrap(), "wb").unwrap();
        imodel_write(&Imod::default(), &mut file, 0).unwrap();
        drop(file);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            i32::from_be_bytes(bytes[152..156].try_into().unwrap()),
            (1 << 13) | (1 << 12) | (1 << 9)
        );
        assert_eq!(bytes.len(), 244);
        assert_eq!(&bytes[240..], b"IEOF");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn binary_imod_roundtrip_keeps_header_contours_sizes_and_meshes() {
        let path =
            std::env::temp_dir().join(format!("imod-rs-imodel-files-{}", std::process::id()));
        let imod = Imod {
            name: name_array("binary fixture"),
            xmax: 10,
            ymax: 20,
            zmax: 30,
            xscale: 1.0,
            yscale: 1.0,
            zscale: 2.0,
            pixsize: 1.5,
            units: -9,
            obj: vec![Iobj {
                name: name_array("object"),
                red: 1.0,
                surfsize: 2,
                cont: vec![Icont {
                    surf: 2,
                    sizes: vec![3.0, 4.0],
                    pts: vec![
                        Ipoint {
                            x: 1.0,
                            y: 2.0,
                            z: 3.0,
                        },
                        Ipoint {
                            x: 4.0,
                            y: 5.0,
                            z: 6.0,
                        },
                    ],
                    ..Icont::default()
                }],
                mesh: vec![Imesh {
                    vert: vec![Ipoint {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    }],
                    list: vec![-1],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_file_write(&imod, &path).unwrap();
        let actual = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(actual.name, imod.name);
        assert_eq!(actual.obj[0].cont[0].pts, imod.obj[0].cont[0].pts);
        assert_eq!(actual.obj[0].cont[0].sizes, imod.obj[0].cont[0].sizes);
        assert_eq!(actual.obj[0].mesh[0].list, imod.obj[0].mesh[0].list);
    }

    #[test]
    fn binary_imod_roundtrip_keeps_model_object_contour_and_mesh_stores() {
        let path = std::env::temp_dir().join(format!("imod-rs-store-{}.mod", std::process::id()));
        let store = super::super::istore::Istore {
            type_: 10,
            flags: 1 << 2,
            index: super::super::istore::StoreUnion::from_i(1),
            value: super::super::istore::StoreUnion::from_f(2.5),
        };
        let model = Imod {
            store: vec![store],
            obj: vec![Iobj {
                store: vec![store],
                cont: vec![Icont {
                    pts: vec![Ipoint::default()],
                    store: vec![store],
                    ..Icont::default()
                }],
                mesh: vec![Imesh {
                    store: vec![store],
                    ..Imesh::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_file_write(&model, &path).unwrap();
        let actual = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for list in [
            &actual.store,
            &actual.obj[0].store,
            &actual.obj[0].cont[0].store,
            &actual.obj[0].mesh[0].store,
        ] {
            assert_eq!(list.len(), 1);
            assert_eq!(list[0].type_, 10);
            assert_eq!((list[0].value.f()), 2.5);
        }
    }

    #[test]
    fn reads_bundled_etomo_fiducial_model() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("IMOD/Etomo/uitestData/BB/BBa_erase.fid");
        let model = imod_read(fixture).unwrap();
        assert!(!model.obj.is_empty());
        assert!(model.obj.iter().any(|object| !object.cont.is_empty()));
    }

    #[test]
    fn extended_binary_chunks_roundtrip() {
        let path = std::env::temp_dir().join(format!("imod-rs-ext-{}", std::process::id()));
        let mut view = Iview::default();
        view.fovy = 33.;
        view.clips.count = 1;
        view.clips.point[0] = Ipoint {
            x: 5.,
            y: 6.,
            z: 7.,
        };
        let imod = Imod {
            flags: (1 << 13) | (1 << 9),
            cview: 1,
            view: vec![Iview::default(), view],
            ref_image: Some(Iref_image {
                ctrans: Ipoint {
                    x: 4.,
                    y: 5.,
                    z: 6.,
                },
                ..Iref_image::default()
            }),
            obj: vec![Iobj {
                mat2: 11,
                mesh_thickness: 15,
                mesh_param: Some(crate::imod::libimod::imesh::MeshParams {
                    flags: 0x1234,
                    passes: 3,
                    cap_skip_nz: 2,
                    tube_diameter: 7.5,
                    cap_skip_zlist: Some(vec![4, 9]),
                    ..Default::default()
                }),
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        imod_file_write(&imod, &path).unwrap();
        let output = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(output.obj[0].mat2, 11);
        assert_eq!(output.obj[0].mesh_thickness, 15);
        let params = output.obj[0].mesh_param.as_ref().unwrap();
        assert_eq!(params.flags, 0x1234);
        assert_eq!(params.passes, 3);
        assert_eq!(params.tube_diameter, 7.5);
        assert_eq!(params.cap_skip_zlist.as_deref(), Some(&[4, 9][..]));
        assert_eq!(output.cview, 1);
        assert_eq!(output.view.len(), 2);
        assert_eq!(
            output.view[1].clips.point[0],
            Ipoint {
                x: 5.,
                y: 6.,
                z: 7.
            }
        );
        assert_eq!(
            output.ref_image.unwrap().ctrans,
            Ipoint {
                x: 4.,
                y: 5.,
                z: 6.
            }
        );
    }

    #[test]
    fn ascii_model_preserves_geometry_view_and_reference_data() {
        let path = std::env::temp_dir().join(format!("imod-rs-ascii-{}", std::process::id()));
        std::fs::write(&path, "imod 1\nmax 10 20 30\nscale 1 2 3\nrefcurscale 2 3 4\nrefcurtrans 5 6 7\nobject 0 1 0\nname old model object\ncolor 0.1 0.2 0.3 4\ncontour 0 2 2\n1 2 3 4\n5 6 7 8\nview 1\nviewfovy 33\nviewtrans 2 3 4\ncurrentview 1\n").unwrap();
        let output = imod_read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            &output.obj[0].name[..b"old model object".len() + 1],
            b"old model object\0"
        );
        assert_eq!(output.obj[0].cont[0].pts.len(), 2);
        assert_eq!(output.obj[0].cont[0].sizes, vec![4., 8.]);
        assert_eq!(output.view.len(), 2);
        assert_eq!(output.view[1].fovy, 33.);
        assert_eq!(
            output.ref_image.unwrap().ctrans,
            Ipoint {
                x: 5.,
                y: 6.,
                z: 7.
            }
        );
    }
}
