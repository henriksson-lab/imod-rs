//! `IMOD/imodutil/imodjoin.c`: join selected objects from IMOD model files.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::{
    CArg, c_format_bytes, imod_backup_file, imod_copyright, imod_prog_name, imod_version,
    replace_file_arg_vec,
};
use crate::imod::libcfshr::parse_params::{exit_error, setExitPrefix};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read};
use crate::imod::libimod::icont::imod_contours_delete;
use crate::imod::libimod::imesh::imod_meshes_delete;
use crate::imod::libimod::imodel::{
    IMODF_FLIPYZ, IMODF_OTRANS_ORIGIN, IMODF_TILTOK, Ipoint, Iref_image, imod_delete_object,
    imod_flip_yz, imod_new_object, imod_trans_from_ref_image, imodel_maxpt,
};
use crate::imod::libimod::imodel_files::{imod_open_file, imod_read, imod_write_file};
use crate::imod::libimod::iobj::imod_object_copy;
use crate::imod::libimod::iview::{
    imod_objview_complete, imod_objview_from_object, imod_view_model_new, imod_view_use,
};

/// Original: `usage` (`imodjoin.c:18`).
pub fn usage() -> ! {
    imod_version(Some("imodjoin"));
    imod_copyright();
    let mut out = ImodFile::Stdout;
    let _ = out.write_all(
        b"Usage: imodjoin [options] model_1 [-o list] model_2 [more models] out_model\n",
    );
    let _ = out.write_all(b"Options (before first model):\n");
    let _ = out
        .write_all(b"\t-o list\tList of objects to take from particular model (default is all)\n");
    let _ = out
        .write_all(b"\t-r list\tList of objects in model 1 to REPLACE with objects from model 2\n");
    let _ = out.write_all(b"\t-c\tChange colors of objects being copied to first model\n");
    let _ = out.write_all(b"\t-d\tModels are from different volumes\n");
    let _ =
        out.write_all(b"\t-i file\tTransform all models to match this image file (implies -d)\n");
    let _ =
        out.write_all(b"\t-s\tIgnore scale differences between models from different volumes\n");
    let _ = out.write_all(b"\t-f\tRetain original flip state of each model\n");
    let _ = out.write_all(b"\t-n\tDo not transforms models at all\n");
    crate::imod::libcfshr::b3dutil::exit(3)
}

/// Original: `parserr` (`imodjoin.c:42`).
pub fn parserr(mod_number: i32) -> ! {
    let _ = ImodFile::Stdout.write_all(
        format!("ERROR: imodjoin - Error parsing object list before model {mod_number}\n")
            .as_bytes(),
    );
    usage()
}
/// Original: `optionerr` (`imodjoin.c:48`).
pub fn optionerr(mod_number: i32) -> ! {
    let _ = ImodFile::Stdout.write_all(
        format!("ERROR: imodjoin - Invalid option before model {mod_number}\n").as_bytes(),
    );
    usage()
}
/// Original: `doublerr` (`imodjoin.c:53`).
pub fn doublerr() -> ! {
    let _ = ImodFile::Stdout
        .write_all(b"ERROR: imodjoin - You cannot use both -o and -r with model 1\n");
    usage()
}
/// Original: `readerr` (`imodjoin.c:59`).
pub fn readerr(mod_number: i32) -> ! {
    let message = format!("Error reading file for model {mod_number}");
    exit_error(message.as_bytes());
}
/// Original: `objerr` (`imodjoin.c:63`).
pub fn objerr(object_number: i32, mod_number: i32) -> ! {
    let message = format!("Invalid object number {object_number} for model {mod_number}");
    exit_error(message.as_bytes());
}

/// Original: `main` (`imodjoin.c:68`).
///
/// Retranslated statement by statement (2026-09-26): the earlier body
/// rebuilt the `-o` reorganisation, the view extension and the object-view
/// completion by hand (compacting object views when a listed object had
/// none), required a literal `-o` where the source reads only the option
/// letter, answered `iarg + 3 >= argc` with `optionerr` instead of `usage`,
/// treated a bare `-` as a model name, and ignored the vector
/// `replaceFileArgVec` returned.
pub fn imodjoin() {
    let args: Vec<String> = crate::imod::libcfshr::b3dutil::program_args();
    let mut argv: Vec<Vec<u8>> = args.iter().map(|a| a.as_bytes().to_vec()).collect();
    let mut argc = argv.len() as i32;
    let arg = |argv: &Vec<Vec<u8>>, i: i32| String::from_utf8_lossy(&argv[i as usize]).into_owned();
    let mut ob: i32;
    let mut nob: i32;
    let mut list1: Vec<i32> = Vec::new();
    let mut list2: Vec<i32>;
    let mut nlist1 = 0_i32;
    let mut nlist2: i32;
    let mut njoin = 1_i32;
    let mut replace = 0;
    let mut first_olist = 0;
    let mut diff_vols = 0;
    let mut flip_state: u32;
    let mut no_ref1 = 0;
    let mut suppress = 0;
    let mut keep_scale = 0;
    let mut keep_flip = 0;
    let mut change_colors = 0;
    let set_max = 1;
    let mut use_ref = Iref_image::default();
    let mut out_ref = Iref_image::default();
    let mut hdata = MrcHeader::default();
    let mut newmax = Ipoint::default();
    let unit_pt = Ipoint {
        x: 1.,
        y: 1.,
        z: 1.,
    };
    let zero_pt = Ipoint {
        x: 0.,
        y: 0.,
        z: 0.,
    };
    let progname = imod_prog_name(args.first().map_or("imodjoin", |a| a.as_str()));
    let prefix = format!("ERROR: {progname} - ");
    setExitPrefix(prefix.as_bytes());

    if argc < 3 {
        usage();
    }

    let mut iarg = 1_i32;
    while iarg < argc {
        if argv[iarg as usize].first() == Some(&b'-') {
            match argv[iarg as usize].get(1).copied().unwrap_or(0) {
                b'h' => usage(),

                b'r' | b'o' => {
                    iarg += 1;
                    // `parselist(argv[++iarg])` past the end is `parselist(NULL)`,
                    // a crash in the source; report it as the parse error.
                    if iarg >= argc {
                        parserr(1);
                    }
                    match parselist(&arg(&argv, iarg)) {
                        Ok(list) => {
                            nlist1 = list.len() as i32;
                            list1 = list;
                        }
                        Err(_) => parserr(1),
                    }
                    if argv[(iarg - 1) as usize][1] == b'r' {
                        replace = 1;
                    } else {
                        first_olist = 1;
                    }
                }

                b'c' => change_colors = 1,

                b'd' => diff_vols = 1,

                b's' => keep_scale = 1,

                b'n' => suppress = 1,

                b'f' => keep_flip = 1,

                b'i' => {
                    iarg += 1;
                    let name = if iarg < argc {
                        arg(&argv, iarg)
                    } else {
                        String::new()
                    };
                    let Some(mut fin) = ImodFile::open(&name, "rb") else {
                        exit_error(&c_format_bytes(
                            "Couldn't open %s",
                            &[CArg::Bytes(name.as_bytes())],
                        ));
                    };

                    if mrc_head_read(&mut fin, &mut hdata) != 0 {
                        exit_error(&c_format_bytes(
                            "Reading header from %s",
                            &[CArg::Bytes(name.as_bytes())],
                        ));
                    }
                    drop(fin);
                    diff_vols = 2;
                }

                _ => optionerr(1),
            }
        } else {
            break;
        }
        iarg += 1;
    }

    if iarg >= argc {
        usage();
    }
    let mut allocated = 0_i32;
    if replace_file_arg_vec(&mut argv, &mut argc, &mut iarg, &mut allocated) != 0 {
        crate::imod::libcfshr::b3dutil::exit(1);
    }

    if replace != 0 && first_olist != 0 {
        doublerr();
    }

    if suppress != 0 && diff_vols != 0 {
        let _ = ImodFile::Stdout
            .write_all(b"ERROR: imodjoin - it makes no sense to use -n with -d or -i\n");
        usage();
    }

    let mut in_model = match imod_read(arg(&argv, iarg)) {
        Ok(model) => model,
        Err(_) => readerr(1),
    };
    iarg += 1;

    /* If model has no refimage, set one up with null transformation */
    if in_model.ref_image.is_none() {
        no_ref1 = 1;
        // `malloc(sizeof(IrefImage))`: `orot` and `oscale` are left
        // uninitialised by the source (`BUGS.md` §2); defined here as the
        // identity, `Iref_image::default()`'s `oscale` (1,1,1) and `orot` 0.
        let mut mod_ref = Iref_image::default();

        let _ = ImodFile::Stdout.write_all(
            b"WARNING: Model 1 has no image reference data; transformations may be wrong\n",
        );
        mod_ref.otrans = zero_pt;
        mod_ref.ctrans = zero_pt;
        mod_ref.crot = zero_pt;
        mod_ref.cscale = unit_pt;
        in_model.ref_image = Some(mod_ref);
        in_model.flags |= IMODF_OTRANS_ORIGIN;
    }

    /* Set up transformations */
    if diff_vols != 0 {
        if diff_vols > 1 {
            /* If reference image file, get the target transformation */
            out_ref.ctrans.x = hdata.xorg;
            out_ref.ctrans.y = hdata.yorg;
            out_ref.ctrans.z = hdata.zorg;
            out_ref.crot.x = hdata.tiltangles[3];
            out_ref.crot.y = hdata.tiltangles[4];
            out_ref.crot.z = hdata.tiltangles[5];
            out_ref.cscale = unit_pt;
            if hdata.xlen != 0. && hdata.mx != 0 {
                out_ref.cscale.x = hdata.xlen / hdata.mx as f32;
            }
            if hdata.ylen != 0. && hdata.my != 0 {
                out_ref.cscale.y = hdata.ylen / hdata.my as f32;
            }
            if hdata.zlen != 0. && hdata.mz != 0 {
                out_ref.cscale.z = hdata.zlen / hdata.mz as f32;
            }

            /* If model had no refs, copy this into it to prevent transform */
            if no_ref1 != 0 {
                let mod_refp = in_model.ref_image.as_mut().unwrap();
                mod_refp.ctrans = out_ref.ctrans;
                mod_refp.otrans = out_ref.ctrans;
                mod_refp.crot = out_ref.crot;
                mod_refp.cscale = out_ref.cscale;
            }
        } else {
            /* No reference image, use model 1 as target with trans set to origin
            if possible */
            let mod_refp = in_model.ref_image.unwrap();
            // `imodjoin.c:216` tests `flags | IMODF_OTRANS_ORIGIN`, which is
            // always true (upstream `|` for `&`); translated as written.
            if (in_model.flags | IMODF_OTRANS_ORIGIN) != 0 {
                out_ref.ctrans = mod_refp.otrans;
            } else {
                out_ref.ctrans = mod_refp.ctrans;
            }
            out_ref.crot = mod_refp.crot;
            out_ref.cscale = mod_refp.cscale;
        }
        let mod_refp = in_model.ref_image.unwrap();

        /* Set up to transform to the reference however it was gotten; namely
        to a full image load coordinates, with rotations ignored */
        use_ref.ctrans = zero_pt;
        use_ref.cscale = out_ref.cscale;
        use_ref.crot = zero_pt;
        use_ref.orot = zero_pt;
        use_ref.oscale = mod_refp.cscale;
        use_ref.otrans = zero_pt;

        /* Use origin information or give warning */
        if (in_model.flags | IMODF_OTRANS_ORIGIN) != 0 {
            use_ref.otrans.x = mod_refp.ctrans.x - mod_refp.otrans.x;
            use_ref.otrans.y = mod_refp.ctrans.y - mod_refp.otrans.y;
            use_ref.otrans.z = mod_refp.ctrans.z - mod_refp.otrans.z;
        } else {
            let _ = ImodFile::Stdout.write_all(
                b"WARNING: Model 1 has no image origin data; transformations may be wrong\n",
            );
        }

        flip_state = if keep_flip != 0 {
            in_model.flags & IMODF_FLIPYZ
        } else {
            0
        };
        if keep_scale != 0 {
            use_ref.cscale = use_ref.oscale;
        }
        imod_trans_from_ref_image(&mut in_model, &use_ref, unit_pt);
        if flip_state != 0 {
            imod_flip_yz(&mut in_model);
            in_model.flags |= IMODF_FLIPYZ;
        }

        /* Set its refimage of the output model, set flags */
        out_ref.otrans = out_ref.ctrans;
        in_model.ref_image = Some(out_ref);
        in_model.flags |= IMODF_OTRANS_ORIGIN | IMODF_TILTOK;
    } else if suppress == 0 {
        /* Models are supposedly from congruent volumes.  Do nothing to model 1,
        it will be the full reference for current transform */
        use_ref = in_model.ref_image.unwrap();
    }

    let origsize = in_model.obj.len() as i32;
    /* If there is a -o list on the first file, reorganize the retained
    objects */
    if nlist1 != 0 && replace == 0 {
        /* Make nlist new objects, then shift all existing objects to top */
        for _ in 0..nlist1 {
            imod_new_object(&mut in_model);
        }
        imod_objview_complete(&mut in_model);
        ob = origsize - 1;
        while ob >= 0 {
            let from = in_model.obj[ob as usize].clone();
            imod_object_copy(&from, &mut in_model.obj[(ob + nlist1) as usize]);
            for iview in 1..in_model.view.len() {
                in_model.view[iview].objview[(ob + nlist1) as usize] =
                    in_model.view[iview].objview[ob as usize].clone();
            }
            ob -= 1;
        }

        /* Now copy all of the selected ones into place */
        for i in 0..nlist1 as usize {
            ob = list1[i] - 1;
            if ob < 0 || ob >= origsize {
                objerr(ob + 1, 1);
            }

            let from = in_model.obj[(ob + nlist1) as usize].clone();
            imod_object_copy(&from, &mut in_model.obj[i]);
            for iview in 1..in_model.view.len() {
                in_model.view[iview].objview[i] =
                    in_model.view[iview].objview[(ob + nlist1) as usize].clone();
            }
        }

        /* Delete extra objects that were not copied - avoid big memory leak */
        ob = origsize + nlist1 - 1;
        while ob >= nlist1 {
            let mut onlist = 0;
            for i in 0..nlist1 as usize {
                if list1[i] - 1 == ob - nlist1 {
                    onlist = 1;
                }
            }
            if onlist == 0 {
                let _ = imod_delete_object(&mut in_model, ob);
            }
            ob -= 1;
        }

        /* Eliminate extra objects by just setting objsize; leak a little */
        in_model.obj.truncate(nlist1 as usize);
        for iview in 1..in_model.view.len() {
            in_model.view[iview].objview.truncate(nlist1 as usize);
        }
    }

    /* process arguments, read model, add objects to first model for one
    or more models */
    loop {
        njoin += 1;

        if iarg + 1 >= argc {
            usage();
        }

        if njoin > 2 && replace != 0 {
            exit_error(b"You cannot use -r with more than 2 input models");
        }

        nlist2 = 0;
        list2 = Vec::new();
        if argv[iarg as usize].first() == Some(&b'-') {
            if iarg + 3 >= argc {
                usage();
            }
            let option = argv[iarg as usize].get(1).copied().unwrap_or(0);
            iarg += 1;
            if option == b'o' {
                match parselist(&arg(&argv, iarg)) {
                    Ok(list) => {
                        nlist2 = list.len() as i32;
                        list2 = list;
                    }
                    Err(_) => parserr(njoin),
                }
                iarg += 1;
            } else {
                optionerr(njoin);
            }
        }

        let mut join_model = match imod_read(arg(&argv, iarg)) {
            Ok(model) => model,
            Err(_) => readerr(njoin),
        };
        iarg += 1;

        /* Set desired flip state to original state or model 1 state */
        flip_state = (if keep_flip != 0 {
            join_model.flags
        } else {
            in_model.flags
        }) & IMODF_FLIPYZ;
        let mod_refp = join_model.ref_image;

        if diff_vols != 0 {
            /* Different volumes: translate to full image load and transform by
            scaling differences only */
            if let Some(mod_refp) = mod_refp {
                use_ref.otrans = zero_pt;

                /* Use origin information or give warning */
                if (join_model.flags | IMODF_OTRANS_ORIGIN) != 0 {
                    use_ref.otrans.x = mod_refp.ctrans.x - mod_refp.otrans.x;
                    use_ref.otrans.y = mod_refp.ctrans.y - mod_refp.otrans.y;
                    use_ref.otrans.z = mod_refp.ctrans.z - mod_refp.otrans.z;
                } else {
                    let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                        "WARNING: Model %d has no image origin data; transformations may be wrong\n",
                        &[CArg::Int(njoin as i64)],
                    ));
                }
                use_ref.oscale = mod_refp.cscale;
                if keep_scale != 0 {
                    use_ref.cscale = use_ref.oscale;
                }
                imod_trans_from_ref_image(&mut join_model, &use_ref, unit_pt);
            } else {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "WARNING: Model %d has no image reference data and will not be transformed\n",
                    &[CArg::Int(njoin as i64)],
                ));
            }
        } else if suppress == 0 {
            /* Same or congruent volumes: if model has refimage set its current
            values to old in the transformation reference and transform;
            otherwise issue a warning */
            if let Some(mod_refp) = mod_refp {
                use_ref.otrans = mod_refp.ctrans;
                use_ref.orot = mod_refp.crot;
                use_ref.oscale = mod_refp.cscale;
                imod_trans_from_ref_image(&mut join_model, &use_ref, unit_pt);
            } else {
                let _ = ImodFile::Stdout.write_all(&c_format_bytes(
                    "WARNING: Model %d has no image reference data and will not be transformed\n",
                    &[CArg::Int(njoin as i64)],
                ));
            }
        }

        /* Set flip to match either the original state or model 1 */
        if flip_state != (join_model.flags & IMODF_FLIPYZ) {
            imod_flip_yz(&mut join_model);
        }

        /* If no list for second model, make simple list of all objects */
        if nlist2 == 0 {
            nlist2 = join_model.obj.len() as i32;
            list2 = (0..nlist2).map(|i| i + 1).collect();
        }

        /* If there are more views in this model, add the extra views to the
        output model */
        if join_model.view.len() > in_model.view.len() {
            while join_model.view.len() > in_model.view.len() {
                imod_view_model_new(&mut in_model);
                let last = in_model.view.len() - 1;
                in_model.view[last] = join_model.view[last].clone();
                in_model.view[last].objview = Vec::new();
            }
            imod_objview_complete(&mut in_model);
        }

        /* Now go through objects in second file, copying selected ones with
        or without replacement */
        for i in 0..nlist2 {
            ob = list2[i as usize] - 1;
            if ob < 0 || ob >= join_model.obj.len() as i32 {
                objerr(ob + 1, njoin);
            }
            if replace != 0 && i < nlist1 {
                nob = list1[i as usize] - 1;
                if nob < 0 || nob >= origsize {
                    objerr(nob + 1, 1);
                }

                /* delete contours and meshes of object being replaced */
                let object = &mut in_model.obj[nob as usize];
                if !object.cont.is_empty() {
                    let contsize = object.cont.len() as i32;
                    imod_contours_delete(&mut object.cont, contsize);
                }
                if !object.mesh.is_empty() {
                    let meshsize = object.mesh.len() as i32;
                    let _ = imod_meshes_delete(Some(std::mem::take(&mut object.mesh)), meshsize);
                }
            } else {
                nob = in_model.obj.len() as i32;
                imod_new_object(&mut in_model);
            }
            let rsave = in_model.obj[nob as usize].red;
            let gsave = in_model.obj[nob as usize].green;
            let bsave = in_model.obj[nob as usize].blue;
            imod_object_copy(
                &join_model.obj[ob as usize],
                &mut in_model.obj[nob as usize],
            );

            /* Restore existing or new object color if selected */
            if change_colors != 0 {
                in_model.obj[nob as usize].red = rsave;
                in_model.obj[nob as usize].green = gsave;
                in_model.obj[nob as usize].blue = bsave;
            }
            imod_objview_complete(&mut in_model);

            /* For each view, if the view exists in the joining model and the
            object view exists for this object, copy object view; otherwise
            copy current object properties to the object view */
            for iview in 1..in_model.view.len() {
                if iview < join_model.view.len()
                    && (ob as usize) < join_model.view[iview].objview.len()
                {
                    in_model.view[iview].objview[nob as usize] =
                        join_model.view[iview].objview[ob as usize].clone();

                    /* If changing colors and color was the same as that of the object,
                    then set to restored color */
                    if change_colors != 0
                        && join_model.view[iview].objview[ob as usize].red
                            == join_model.obj[ob as usize].red
                        && join_model.view[iview].objview[ob as usize].green
                            == join_model.obj[ob as usize].green
                        && join_model.view[iview].objview[ob as usize].blue
                            == join_model.obj[ob as usize].blue
                    {
                        in_model.view[iview].objview[nob as usize].red = rsave;
                        in_model.view[iview].objview[nob as usize].green = gsave;
                        in_model.view[iview].objview[nob as usize].blue = bsave;
                    }
                } else {
                    let object = in_model.obj[nob as usize].clone();
                    imod_objview_from_object(
                        &object,
                        &mut in_model.view[iview].objview[nob as usize],
                    );
                }
            }
        }

        /* We can't free the model because data were transferred from it.
        Delete extra objects that were not copied - avoid big memory leak */
        ob = join_model.obj.len() as i32 - 1;
        while ob >= 0 {
            let mut onlist = 0;
            for i in 0..nlist2 as usize {
                if list2[i] - 1 == ob {
                    onlist = 1;
                }
            }
            if onlist == 0 {
                let _ = imod_delete_object(&mut join_model, ob);
            }
            ob -= 1;
        }

        drop(list2);
        if !(iarg + 1 < argc) {
            break;
        }
    }

    /* Synchronize object data to current view values if there are any
    real views*/
    if in_model.cview == 0 && in_model.view.len() > 1 {
        in_model.cview = 1;
    }
    if in_model.cview != 0 {
        imod_view_use(&mut in_model);
    }

    /* set current indexes to -1 to avoid problems */
    in_model.cindex.point = -1;
    in_model.cindex.contour = -1;
    in_model.cindex.object = 0;

    /* Set max of model big enough to long the whole thing */
    if set_max != 0 {
        imodel_maxpt(&in_model, &mut newmax);
        // `B3DMAX(int, float)` is `a > b ? a : b` evaluated in float, then
        // stored back into the int member.
        let x = in_model.xmax as f32;
        in_model.xmax = (if x > newmax.x { x } else { newmax.x }) as i32;
        let y = in_model.ymax as f32;
        in_model.ymax = (if y > newmax.y { y } else { newmax.y }) as i32;
        let z = in_model.zmax as f32;
        in_model.zmax = (if z > newmax.z { z } else { newmax.z }) as i32;
    }

    let output = arg(&argv, argc - 1);
    imod_backup_file(&output);
    let mut fout = match imod_open_file(&output, "wb", &mut in_model) {
        Ok(file) => file,
        Err(_) => exit_error(b"Fatal error opening new model"),
    };
    let _ = imod_write_file(&in_model, &mut fout);
    drop(fout);
    crate::imod::libcfshr::b3dutil::exit(0);
}
