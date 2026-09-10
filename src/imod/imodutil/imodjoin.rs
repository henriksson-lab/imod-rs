//! `IMOD/imodutil/imodjoin.c`: join selected objects from IMOD model files.

use std::ffi::CString;
use std::io::Write;
use std::path::Path;
use std::{env, fs};

use crate::imod::libcfshr::b3dutil::{imod_copyright, imod_version, replace_file_arg_vec};
use crate::imod::libcfshr::parselist::parselist;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read};
use crate::imod::libimod::imodel::{
    IMODF_FLIPYZ, IMODF_OTRANS_ORIGIN, IMODF_TILTOK, Iobj, Iobjview, Ipoint, Iref_image,
    imod_flip_yz, imod_trans_from_ref_image,
};
use crate::imod::libimod::imodel_files::{imod_file_write, imod_read};

/// Original: `usage` (`imodjoin.c:18`).
pub fn usage() -> ! {
    unsafe { imod_version(c"imodjoin".as_ptr()) };
    imod_copyright();
    unsafe { libc::fflush(std::ptr::null_mut()) };
    println!("Usage: imodjoin [options] model_1 [-o list] model_2 [more models] out_model");
    println!("Options (before first model):");
    println!("\t-o list\tList of objects to take from particular model (default is all)");
    println!("\t-r list\tList of objects in model 1 to REPLACE with objects from model 2");
    println!("\t-c\tChange colors of objects being copied to first model");
    println!("\t-d\tModels are from different volumes");
    println!("\t-i file\tTransform all models to match this image file (implies -d)");
    println!("\t-s\tIgnore scale differences between models from different volumes");
    println!("\t-f\tRetain original flip state of each model");
    println!("\t-n\tDo not transforms models at all");
    let _ = std::io::stdout().flush();
    std::process::exit(3)
}

/// Original: `parserr` (`imodjoin.c:42`).
pub fn parserr(mod_number: i32) -> ! {
    println!("ERROR: imodjoin - Error parsing object list before model {mod_number}");
    let _ = std::io::stdout().flush();
    usage()
}
/// Original: `optionerr` (`imodjoin.c:48`).
pub fn optionerr(mod_number: i32) -> ! {
    println!("ERROR: imodjoin - Invalid option before model {mod_number}");
    let _ = std::io::stdout().flush();
    usage()
}
/// Original: `doublerr` (`imodjoin.c:53`).
pub fn doublerr() -> ! {
    println!("ERROR: imodjoin - You cannot use both -o and -r with model 1");
    let _ = std::io::stdout().flush();
    usage()
}
/// Original: `readerr` (`imodjoin.c:59`).
pub fn readerr(mod_number: i32) -> ! {
    eprintln!("ERROR: imodjoin - Error reading file for model {mod_number}");
    std::process::exit(1)
}
/// Original: `objerr` (`imodjoin.c:63`).
pub fn objerr(object_number: i32, mod_number: i32) -> ! {
    eprintln!("ERROR: imodjoin - Invalid object number {object_number} for model {mod_number}");
    std::process::exit(1)
}

/// Original: `main` (`imodjoin.c:68`).
pub fn imodjoin() {
    let argv: Vec<String> = env::args().collect();
    if argv.len() < 3 {
        usage();
    }
    let mut list1 = Vec::<usize>::new();
    let mut replace = false;
    let mut first_olist = false;
    let mut diff_vols = false;
    let mut suppress = false;
    let mut keep_scale = false;
    let mut keep_flip = false;
    let mut change_colors = false;
    let mut image_reference = None::<Iref_image>;
    let mut iarg = 1;
    while iarg < argv.len() && argv[iarg].starts_with('-') && argv[iarg] != "-" {
        let option = argv[iarg].as_bytes().get(1).copied().unwrap_or_default() as char;
        match option {
            'h' => usage(),
            'r' | 'o' => {
                iarg += 1;
                let Some(text) = argv.get(iarg) else {
                    parserr(1)
                };
                let text = CString::new(text.as_str()).unwrap_or_else(|_| parserr(1));
                let mut nlist1 = 0_i32;
                let parsed = unsafe { parselist(text.as_ptr(), &mut nlist1) };
                if parsed.is_null() {
                    parserr(1);
                }
                list1 = unsafe { std::slice::from_raw_parts(parsed, nlist1 as usize) }
                    .iter()
                    .map(|&value| value as usize)
                    .collect();
                unsafe { libc::free(parsed.cast()) };
                if option == 'r' {
                    replace = true;
                } else {
                    first_olist = true;
                }
            }
            'c' => change_colors = true,
            'd' => diff_vols = true,
            's' => keep_scale = true,
            'n' => suppress = true,
            'f' => keep_flip = true,
            'i' => {
                iarg += 1;
                let Some(image) = argv.get(iarg) else {
                    optionerr(1)
                };
                let image_name = CString::new(image.as_str()).unwrap_or_else(|_| optionerr(1));
                let fin = unsafe { libc::fopen(image_name.as_ptr(), c"rb".as_ptr()) };
                if fin.is_null() {
                    eprintln!("ERROR: imodjoin - Couldn't open {image}");
                    std::process::exit(1);
                }
                let mut hdata = unsafe { std::mem::zeroed::<MrcHeader>() };
                if unsafe { mrc_head_read(fin, &mut hdata) } != 0 {
                    unsafe { libc::fclose(fin) };
                    eprintln!("ERROR: imodjoin - Reading header from {image}");
                    std::process::exit(1);
                }
                unsafe { libc::fclose(fin) };
                let mut reference = Iref_image::default();
                reference.ctrans = Ipoint {
                    x: hdata.xorg,
                    y: hdata.yorg,
                    z: hdata.zorg,
                };
                reference.crot = Ipoint {
                    x: hdata.tiltangles[3],
                    y: hdata.tiltangles[4],
                    z: hdata.tiltangles[5],
                };
                reference.cscale = Ipoint {
                    x: if hdata.xlen != 0. && hdata.mx != 0 {
                        hdata.xlen / hdata.mx as f32
                    } else {
                        1.
                    },
                    y: if hdata.ylen != 0. && hdata.my != 0 {
                        hdata.ylen / hdata.my as f32
                    } else {
                        1.
                    },
                    z: if hdata.zlen != 0. && hdata.mz != 0 {
                        hdata.zlen / hdata.mz as f32
                    } else {
                        1.
                    },
                };
                image_reference = Some(reference);
                diff_vols = true;
            }
            _ => optionerr(1),
        };
        iarg += 1;
    }
    if iarg >= argv.len() {
        usage();
    }
    let argument_strings: Vec<CString> = argv
        .iter()
        .map(|argument| CString::new(argument.as_str()).unwrap_or_else(|_| usage()))
        .collect();
    let argument_pointers: Vec<*const libc::c_char> = argument_strings
        .iter()
        .map(|argument| argument.as_ptr())
        .collect();
    let mut argument_vector = argument_pointers.as_ptr();
    let mut argument_count = argv.len() as i32;
    let mut first_argument = iarg as i32;
    let mut allocated = 0_i32;
    if unsafe {
        replace_file_arg_vec(
            &mut argument_vector,
            &mut argument_count,
            &mut first_argument,
            &mut allocated,
        )
    } != 0
    {
        std::process::exit(1);
    }
    if replace && first_olist {
        doublerr();
    }
    if suppress && diff_vols {
        println!("ERROR: imodjoin - it makes no sense to use -n with -d or -i");
        usage();
    }
    let mut in_model = imod_read(&argv[iarg]).unwrap_or_else(|_| readerr(1));
    iarg += 1;
    let original_size = in_model.obj.len();
    let mut no_ref1 = false;
    if in_model.ref_image.is_none() {
        no_ref1 = true;
        let mut reference = Iref_image::default();
        reference.otrans = Ipoint::default();
        reference.ctrans = Ipoint::default();
        reference.crot = Ipoint::default();
        reference.cscale = Ipoint {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        in_model.ref_image = Some(reference);
        println!(
            "WARNING: imodjoin - Model 1 has no image reference data; transformations may be wrong"
        );
        in_model.flags |= IMODF_OTRANS_ORIGIN;
    }
    let mut use_ref = Iref_image::default();
    if diff_vols {
        let mut mod_ref = in_model.ref_image.unwrap();
        let mut out_ref = Iref_image::default();
        if let Some(reference) = image_reference {
            out_ref.ctrans = reference.ctrans;
            out_ref.crot = reference.crot;
            out_ref.cscale = reference.cscale;
            if no_ref1 {
                let model_ref = in_model.ref_image.as_mut().unwrap();
                model_ref.ctrans = out_ref.ctrans;
                model_ref.otrans = out_ref.ctrans;
                model_ref.crot = out_ref.crot;
                model_ref.cscale = out_ref.cscale;
                mod_ref = *model_ref;
            }
        } else {
            out_ref.ctrans = mod_ref.otrans;
            out_ref.crot = mod_ref.crot;
            out_ref.cscale = mod_ref.cscale;
        }
        use_ref.ctrans = Ipoint::default();
        use_ref.cscale = out_ref.cscale;
        use_ref.crot = Ipoint::default();
        use_ref.orot = Ipoint::default();
        use_ref.oscale = mod_ref.cscale;
        use_ref.otrans = Ipoint::default();
        use_ref.otrans.x = mod_ref.ctrans.x - mod_ref.otrans.x;
        use_ref.otrans.y = mod_ref.ctrans.y - mod_ref.otrans.y;
        use_ref.otrans.z = mod_ref.ctrans.z - mod_ref.otrans.z;
        let flip_state = if keep_flip {
            in_model.flags & IMODF_FLIPYZ
        } else {
            0
        };
        if keep_scale {
            use_ref.cscale = use_ref.oscale;
        }
        let _ = imod_trans_from_ref_image(
            &mut in_model,
            &use_ref,
            Ipoint {
                x: 1.,
                y: 1.,
                z: 1.,
            },
        );
        if flip_state != 0 {
            imod_flip_yz(&mut in_model);
            in_model.flags |= IMODF_FLIPYZ;
        }
        out_ref.otrans = out_ref.ctrans;
        in_model.ref_image = Some(out_ref);
        in_model.flags |= IMODF_OTRANS_ORIGIN | IMODF_TILTOK;
    } else if !suppress {
        use_ref = in_model.ref_image.unwrap();
    }
    if !list1.is_empty() && !replace {
        let mut objects = Vec::with_capacity(list1.len());
        for &number in &list1 {
            let index = number.checked_sub(1).unwrap_or(usize::MAX);
            if index >= original_size {
                objerr(number as i32, 1);
            }
            objects.push(in_model.obj[index].clone());
        }
        in_model.obj = objects;
        for iview in 1..in_model.view.len() {
            let mut object_views = Vec::with_capacity(list1.len());
            for &number in &list1 {
                let index = number - 1;
                if index < in_model.view[iview].objview.len() {
                    object_views.push(in_model.view[iview].objview[index].clone());
                }
            }
            in_model.view[iview].objview = object_views;
        }
    }
    let mut njoin = 1;
    loop {
        // The C source uses a do/while loop: a second input model and an
        // output are mandatory after model 1.  Its first loop gate must run
        // before considering the last argument as an output name.
        if iarg + 1 >= argv.len() {
            usage();
        }
        njoin += 1;
        if njoin > 2 && replace {
            eprintln!("ERROR: imodjoin - You cannot use -r with more than 2 input models");
            std::process::exit(1);
        }
        let mut list2 = Vec::<usize>::new();
        if argv[iarg].starts_with('-') {
            if !argv[iarg].starts_with("-o") || iarg + 1 >= argv.len() {
                optionerr(njoin);
            }
            iarg += 1;
            let text = CString::new(argv[iarg].as_str()).unwrap_or_else(|_| parserr(njoin));
            let mut nlist2 = 0_i32;
            let parsed = unsafe { parselist(text.as_ptr(), &mut nlist2) };
            if parsed.is_null() {
                parserr(njoin);
            }
            list2 = unsafe { std::slice::from_raw_parts(parsed, nlist2 as usize) }
                .iter()
                .map(|&value| value as usize)
                .collect();
            unsafe { libc::free(parsed.cast()) };
            iarg += 1;
        }
        let mut join_model = imod_read(&argv[iarg]).unwrap_or_else(|_| readerr(njoin));
        iarg += 1;
        if !suppress {
            if let Some(join_ref) = join_model.ref_image {
                if diff_vols {
                    use_ref.otrans = Ipoint::default();
                    use_ref.otrans.x = join_ref.ctrans.x - join_ref.otrans.x;
                    use_ref.otrans.y = join_ref.ctrans.y - join_ref.otrans.y;
                    use_ref.otrans.z = join_ref.ctrans.z - join_ref.otrans.z;
                    use_ref.oscale = join_ref.cscale;
                } else {
                    use_ref.otrans = join_ref.ctrans;
                    use_ref.orot = join_ref.crot;
                    use_ref.oscale = join_ref.cscale;
                }
                if keep_scale {
                    use_ref.cscale = use_ref.oscale;
                }
                let wanted_flip = if keep_flip {
                    join_model.flags & IMODF_FLIPYZ
                } else {
                    in_model.flags & IMODF_FLIPYZ
                };
                let _ = imod_trans_from_ref_image(
                    &mut join_model,
                    &use_ref,
                    Ipoint {
                        x: 1.,
                        y: 1.,
                        z: 1.,
                    },
                );
                if wanted_flip != join_model.flags & IMODF_FLIPYZ {
                    imod_flip_yz(&mut join_model);
                    join_model.flags ^= IMODF_FLIPYZ;
                }
            } else {
                println!(
                    "WARNING: imodjoin - Model {njoin} has no image reference data and will not be transformed"
                );
            }
        }
        if list2.is_empty() {
            list2 = (1..=join_model.obj.len()).collect();
        }
        if join_model.view.len() > in_model.view.len() {
            while join_model.view.len() > in_model.view.len() {
                let iview = in_model.view.len();
                let mut view = join_model.view[iview].clone();
                view.objview.clear();
                in_model.view.push(view);
            }
        }
        for iview in 1..in_model.view.len() {
            while in_model.view[iview].objview.len() < in_model.obj.len() {
                let object = &in_model.obj[in_model.view[iview].objview.len()];
                in_model.view[iview].objview.push(Iobjview {
                    flags: object.flags,
                    red: object.red,
                    green: object.green,
                    blue: object.blue,
                    pdrawsize: object.pdrawsize,
                    linewidth: object.linewidth,
                    linesty: object.linesty,
                    trans: object.trans,
                    clips: object.clips.clone(),
                    ambient: object.ambient,
                    diffuse: object.diffuse,
                    specular: object.specular,
                    shininess: object.shininess,
                    fillred: object.fillred,
                    fillgreen: object.fillgreen,
                    fillblue: object.fillblue,
                    quality: object.quality,
                    mat2: object.mat2,
                    valblack: object.valblack,
                    valwhite: object.valwhite,
                    matflags2: object.matflags2,
                    mesh_thickness: object.mesh_thickness,
                });
            }
        }
        for (ordinal, &number) in list2.iter().enumerate() {
            let source_index = number.checked_sub(1).unwrap_or(usize::MAX);
            let Some(source) = join_model.obj.get(source_index) else {
                objerr(number as i32, njoin)
            };
            let destination = if replace && ordinal < list1.len() {
                let index = list1[ordinal].checked_sub(1).unwrap_or(usize::MAX);
                if index >= original_size {
                    objerr(list1[ordinal] as i32, 1);
                }
                index
            } else {
                in_model.obj.push(Iobj::default());
                in_model.obj.len() - 1
            };
            let (red, green, blue) = (
                in_model.obj[destination].red,
                in_model.obj[destination].green,
                in_model.obj[destination].blue,
            );
            in_model.obj[destination] = source.clone();
            if change_colors {
                in_model.obj[destination].red = red;
                in_model.obj[destination].green = green;
                in_model.obj[destination].blue = blue;
            }
            for iview in 1..in_model.view.len() {
                while in_model.view[iview].objview.len() <= destination {
                    let object = &in_model.obj[in_model.view[iview].objview.len()];
                    in_model.view[iview].objview.push(Iobjview {
                        flags: object.flags,
                        red: object.red,
                        green: object.green,
                        blue: object.blue,
                        pdrawsize: object.pdrawsize,
                        linewidth: object.linewidth,
                        linesty: object.linesty,
                        trans: object.trans,
                        clips: object.clips.clone(),
                        ambient: object.ambient,
                        diffuse: object.diffuse,
                        specular: object.specular,
                        shininess: object.shininess,
                        fillred: object.fillred,
                        fillgreen: object.fillgreen,
                        fillblue: object.fillblue,
                        quality: object.quality,
                        mat2: object.mat2,
                        valblack: object.valblack,
                        valwhite: object.valwhite,
                        matflags2: object.matflags2,
                        mesh_thickness: object.mesh_thickness,
                    });
                }
                if iview < join_model.view.len()
                    && source_index < join_model.view[iview].objview.len()
                {
                    in_model.view[iview].objview[destination] =
                        join_model.view[iview].objview[source_index].clone();
                    if change_colors
                        && join_model.view[iview].objview[source_index].red
                            == join_model.obj[source_index].red
                        && join_model.view[iview].objview[source_index].green
                            == join_model.obj[source_index].green
                        && join_model.view[iview].objview[source_index].blue
                            == join_model.obj[source_index].blue
                    {
                        in_model.view[iview].objview[destination].red = red;
                        in_model.view[iview].objview[destination].green = green;
                        in_model.view[iview].objview[destination].blue = blue;
                    }
                } else {
                    let object = &in_model.obj[destination];
                    in_model.view[iview].objview[destination] = Iobjview {
                        flags: object.flags,
                        red: object.red,
                        green: object.green,
                        blue: object.blue,
                        pdrawsize: object.pdrawsize,
                        linewidth: object.linewidth,
                        linesty: object.linesty,
                        trans: object.trans,
                        clips: object.clips.clone(),
                        ambient: object.ambient,
                        diffuse: object.diffuse,
                        specular: object.specular,
                        shininess: object.shininess,
                        fillred: object.fillred,
                        fillgreen: object.fillgreen,
                        fillblue: object.fillblue,
                        quality: object.quality,
                        mat2: object.mat2,
                        valblack: object.valblack,
                        valwhite: object.valwhite,
                        matflags2: object.matflags2,
                        mesh_thickness: object.mesh_thickness,
                    };
                }
            }
        }
        if iarg + 1 >= argv.len() {
            break;
        }
    }
    if in_model.cview == 0 && in_model.view.len() > 1 {
        in_model.cview = 1;
    }
    if in_model.cview > 0 && (in_model.cview as usize) < in_model.view.len() {
        let label = in_model.view[0].label;
        let default_object_views = in_model.view[0].objview.clone();
        let current_view = in_model.view[in_model.cview as usize].clone();
        in_model.view[0] = current_view.clone();
        in_model.view[0].label = label;
        in_model.view[0].objview = default_object_views;
        for object_number in 0..in_model.obj.len().min(current_view.objview.len()) {
            let object_view = &current_view.objview[object_number];
            let object = &mut in_model.obj[object_number];
            object.flags = object_view.flags;
            object.red = object_view.red;
            object.green = object_view.green;
            object.blue = object_view.blue;
            object.pdrawsize = object_view.pdrawsize;
            object.linewidth = object_view.linewidth;
            object.linesty = object_view.linesty;
            object.trans = object_view.trans;
            object.clips = object_view.clips.clone();
            object.ambient = object_view.ambient;
            object.diffuse = object_view.diffuse;
            object.specular = object_view.specular;
            object.shininess = object_view.shininess;
            object.fillred = object_view.fillred;
            object.fillgreen = object_view.fillgreen;
            object.fillblue = object_view.fillblue;
            object.quality = object_view.quality;
            object.mat2 = object_view.mat2;
            object.valblack = object_view.valblack;
            object.valwhite = object_view.valwhite;
            object.matflags2 = object_view.matflags2;
            object.mesh_thickness = object_view.mesh_thickness;
        }
    }
    in_model.cindex.point = -1;
    in_model.cindex.contour = -1;
    in_model.cindex.object = 0;
    let mut max = Ipoint::default();
    for object in &in_model.obj {
        for contour in &object.cont {
            for point in &contour.pts {
                max.x = max.x.max(point.x);
                max.y = max.y.max(point.y);
                max.z = max.z.max(point.z);
            }
        }
        for mesh in &object.mesh {
            for point in &mesh.vert {
                max.x = max.x.max(point.x);
                max.y = max.y.max(point.y);
                max.z = max.z.max(point.z);
            }
        }
    }
    in_model.xmax = in_model.xmax.max(max.x as i32);
    in_model.ymax = in_model.ymax.max(max.y as i32);
    in_model.zmax = in_model.zmax.max(max.z as i32);
    let output = &argv[argv.len() - 1];
    let backup = format!("{output}~");
    if Path::new(output).exists() {
        let _ = fs::remove_file(&backup);
        let _ = fs::rename(output, &backup);
    }
    imod_file_write(&in_model, output).unwrap_or_else(|_| {
        eprintln!("ERROR: imodjoin - Fatal error opening new model");
        std::process::exit(1)
    });
}
