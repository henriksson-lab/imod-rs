//! `IMOD/imodutil/wmod2imod.c`: convert a legacy ASCII WIMP model to IMOD.

use std::env;
use std::ffi::CString;
use std::fs::File;

use crate::imod::libcfshr::b3dutil::{ImodFile, fgetline};
use crate::imod::libimod::imodel::{
    IMOD_OBJFLAG_OPEN, Imod, Ipoint, imod_new_contour, imod_new_object,
};
use crate::imod::libimod::imodel_files::imod_write;
use crate::imod::libimod::ipoint::imod_point_add;

/// Original: `main` (`wmod2imod.c:21`).
pub fn wmod2imod() {
    let argv: Vec<String> = env::args().collect();
    if argv.len() < 2 {
        println!("wmod2imod version 1.0 usage:");
        println!("wmod2imod [-z scale] [wmod filename] [imod filename]");
        std::process::exit(1);
    }
    let mut xscale = 1.0_f32;
    let mut yscale = 1.0_f32;
    let mut zscale = 1.0_f32;
    let mut i = 1_usize;
    while i < argv.len() {
        if argv[i].starts_with('-') {
            let option = argv[i].as_bytes().get(1).copied();
            match option {
                Some(b'x') => {
                    i += 1;
                    let value =
                        CString::new(argv.get(i).map(String::as_str).unwrap_or("")).unwrap();
                    unsafe { libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &mut xscale) };
                }
                Some(b'y') => {
                    i += 1;
                    let value =
                        CString::new(argv.get(i).map(String::as_str).unwrap_or("")).unwrap();
                    unsafe { libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &mut yscale) };
                }
                Some(b'z') => {
                    i += 1;
                    let value =
                        CString::new(argv.get(i).map(String::as_str).unwrap_or("")).unwrap();
                    unsafe { libc::sscanf(value.as_ptr(), c"%f".as_ptr(), &mut zscale) };
                }
                _ => {}
            }
            i += 1;
        } else {
            break;
        }
    }
    if i > argv.len().saturating_sub(2) {
        println!("i = {i}");
        println!("wmod2imod version 0.9 usage:");
        println!("wmod2imod [-z scale] [wmod filename] [imod filename]");
        std::process::exit(1);
    }
    let fin = ImodFile::open(&argv[i], "r");
    let Some(mut fin) = fin else {
        // C increments i before this diagnostic, so it prints the output name.
        eprintln!(
            "Couldn't open {}",
            argv.get(i + 1).unwrap_or(&argv[i]).as_str()
        );
        std::process::exit(3)
    };
    i += 1;
    let output = &argv[i];
    let Some(mut fout) = ImodFile::open(output, "wb") else {
        eprintln!("Couldn't open {output}");
        std::process::exit(3)
    };
    let mut model = imod_from_wmod(&mut fin).unwrap_or_else(|| {
        eprintln!("Error reading imod file.");
        std::process::exit(3)
    });
    // Preserve the source assignments, including its historical -x typo.
    model.zscale = xscale;
    model.yscale = yscale;
    model.zscale = zscale;
    imod_write(&model, &mut fout).unwrap_or_else(|_| std::process::exit(3));
}

/// Original: `imod_from_wmod` (`wmod2imod.c:122`).
pub fn imod_from_wmod(fin: &mut ImodFile) -> Option<Imod> {
    const MAXLINE: i32 = 128;
    const MAXOBJ: usize = 256;
    /// Original: `Wmod_Colors` (`wmod2imod.c:100`).
    const WMOD_COLORS: [[f32; 3]; 9] = [
        [0.90, 0.82, 0.37],
        [0.54, 0.51, 0.01],
        [0.94, 0.49, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let cont_string = c"Object #:";
    let mut line = [0_u8; MAXLINE as usize];
    let mut objlookup = [0_i32; MAXOBJ];
    let mut nobj = 0_i32;
    let mut display_switch = 0_i32;
    let mut points = 0_i32;
    // `struct Mod_Point point` has function scope in the source, so a point
    // line that fails to scan re-adds the previously scanned coordinates.
    let mut point = Ipoint::default();
    unsafe {
        loop {
            let len = fgetline(fin, &mut line, MAXLINE);
            if len < 0 {
                break;
            }
            if len == 0 {
                continue;
            }
            let mut tline = std::ptr::null::<std::ffi::c_char>();
            let mut i = 0_usize;
            while line[i] != 0 {
                if line[i] == b'O' {
                    tline = line.as_ptr().add(i).cast();
                    break;
                }
                i += 1;
            }
            if tline.is_null() {
                continue;
            }
            // `substr` (`imodel_from.c:207`) compares the first strlen(ls) bytes.
            if libc::strncmp(tline, cont_string.as_ptr(), cont_string.count_bytes()) != 0 {
                continue;
            }
            fgetline(fin, &mut line, MAXLINE);
            libc::sscanf(
                line.as_ptr().cast(),
                c"%*s %*s %*s %d".as_ptr(),
                &mut points,
            );
            fgetline(fin, &mut line, MAXLINE);
            libc::sscanf(
                line.as_ptr().cast(),
                c"%*s %*s %d".as_ptr(),
                &mut display_switch,
            );
            if display_switch >= 0 && (display_switch as usize) < MAXOBJ {
                objlookup[display_switch as usize] = 1;
            }
        }
    }
    let mut model = Imod::default();
    // `wmod2imod.c:171` obtains this through `imodNew`, whose `imodDefault`
    // state is written into the resulting binary model.
    // `imodel.c:48-54` copies 13 bytes of the literal and terminates at [13],
    // leaving the rest of the IMOD_STRSIZE array as allocated.
    let newmodname = b"IMOD-NewModel";
    for i in 0..13 {
        model.name[i] = newmodname[i];
    }
    model.name[13] = 0x00;
    model.flags = (1 << 11) | (1 << 10);
    model.drawmode = 1;
    model.mousemode = 2;
    model.whitelevel = 255;
    model.xscale = 1.0;
    model.yscale = 1.0;
    model.zscale = 1.0;
    model.res = 3;
    model.thresh = 128;
    model.pixsize = 1.0;
    model.xmax = 1;
    model.ymax = 1;
    model.zmax = 1;
    for wimp_number in 0..MAXOBJ {
        if objlookup[wimp_number] != 0 {
            objlookup[wimp_number] = nobj;
            if imod_new_object(&mut model) != 0 {
                return None;
            }
            let object = model.obj.get_mut(nobj as usize)?;
            if wimp_number >= 247 {
                object.red = WMOD_COLORS[wimp_number - 247][0];
                object.green = WMOD_COLORS[wimp_number - 247][1];
                object.blue = WMOD_COLORS[wimp_number - 247][2];
            } else {
                object.red = wimp_number as f32 / 255.0_f32;
                object.green = wimp_number as f32 / 255.0_f32;
                object.blue = wimp_number as f32 / 255.0_f32;
            }
            // `wmod2imod.c:184`: `sprintf(mod->obj[nobj].name, "Wimp no. %d", i)`.
            let name = crate::imod::libcfshr::b3dutil::c_format(
                "Wimp no. %d",
                &[crate::imod::libcfshr::b3dutil::CArg::Int(
                    wimp_number as i64,
                )],
            );
            object.name[..name.len()].copy_from_slice(name.as_bytes());
            object.name[name.len()] = 0;
            nobj += 1;
        } else {
            objlookup[wimp_number] = -1;
        }
    }
    unsafe {
        use std::io::Seek;
        let _ = fin.rewind();
        loop {
            let len = fgetline(fin, &mut line, MAXLINE);
            if len < 0 {
                break;
            }
            if len == 0 {
                continue;
            }
            let mut tline = std::ptr::null::<std::ffi::c_char>();
            let mut i = 0_usize;
            while line[i] != 0 {
                if line[i] == b'O' {
                    tline = line.as_ptr().add(i).cast();
                    break;
                }
                i += 1;
            }
            if tline.is_null() {
                continue;
            }
            if libc::strncmp(tline, cont_string.as_ptr(), cont_string.count_bytes()) != 0 {
                continue;
            }
            fgetline(fin, &mut line, MAXLINE);
            libc::sscanf(
                line.as_ptr().cast(),
                c"%*s %*s %*s %d".as_ptr(),
                &mut points,
            );
            fgetline(fin, &mut line, MAXLINE);
            libc::sscanf(
                line.as_ptr().cast(),
                c"%*s %*s %d".as_ptr(),
                &mut display_switch,
            );
            fgetline(fin, &mut line, MAXLINE);
            model.cindex.object = if display_switch >= 0 && (display_switch as usize) < MAXOBJ {
                objlookup[display_switch as usize]
            } else {
                0
            };
            if model.cindex.object < 0 {
                model.cindex.object = 0;
            }
            let object_index = model.cindex.object as usize;
            model.cindex.contour = model.obj.get(object_index)?.cont.len() as i32 - 1;
            if imod_new_contour(&mut model) != 0 {
                return None;
            }
            let contour_index = model.cindex.contour as usize;
            for i in 0..points {
                fgetline(fin, &mut line, MAXLINE);
                libc::sscanf(
                    line.as_ptr().cast(),
                    c"%*d %f %f %f".as_ptr(),
                    &mut point.x,
                    &mut point.y,
                    &mut point.z,
                );
                let contour = model
                    .obj
                    .get_mut(object_index)?
                    .cont
                    .get_mut(contour_index)?;
                if imod_point_add(contour, Some(point), i) == 0 {
                    return None;
                }
            }
            // `wmod2imod.c:246` re-asserts the contour size, which the
            // `points` calls to `imodPointAdd` above have already produced.
            for object in &mut model.obj {
                if !object.cont.is_empty() {
                    if object.cont[0].pts.len() < 2
                        || object.cont[0].pts[0].z != object.cont[0].pts[1].z
                    {
                        object.flags |= IMOD_OBJFLAG_OPEN;
                    }
                }
            }
        }
    }
    // `imodDeleteObject` (`imodel.c`) drops every object that got no contour.
    model.obj.retain(|object| !object.cont.is_empty());
    Some(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libimod::imodel::{ICONT_WILD, Icont, Iobj};
    use crate::imod::libimod::imodel_files::{imod_file_write, imod_read};
    use crate::imod::libimod::imodel_to::imod_to_wmod;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn wimp_to_imod_roundtrip_keeps_real_wimp_geometry_colors_and_open_flag() {
        let root = std::env::temp_dir().join(format!("imod-rs-wmod2imod-{}", std::process::id()));
        let wimp = root.with_extension("wimp");
        let imod = root.with_extension("mod");
        let source = Imod {
            obj: vec![Iobj {
                cont: vec![Icont {
                    pts: vec![
                        Ipoint {
                            x: 1.,
                            y: 2.,
                            z: 3.,
                        },
                        Ipoint {
                            x: 4.,
                            y: 5.,
                            z: 6.,
                        },
                    ],
                    ..Icont::default()
                }],
                ..Iobj::default()
            }],
            ..Imod::default()
        };
        let mut wimp_file = File::create(&wimp).unwrap();
        imod_to_wmod(&source, &mut wimp_file, "fixture.wimp").unwrap();
        wimp_file.flush().unwrap();
        let mut wimp_file = ImodFile::open(wimp.to_str().unwrap(), "r").unwrap();
        let converted = imod_from_wmod(&mut wimp_file).unwrap();
        drop(wimp_file);
        assert_eq!(converted.obj.len(), 1);
        assert_eq!(
            &converted.obj[0].name[..b"Wimp no. 247".len() + 1],
            b"Wimp no. 247\0"
        );
        assert_eq!(
            (
                converted.obj[0].red,
                converted.obj[0].green,
                converted.obj[0].blue
            ),
            (0.90, 0.82, 0.37)
        );
        assert_eq!(converted.obj[0].cont[0].pts, source.obj[0].cont[0].pts);
        assert_eq!(
            (
                converted.cindex.object,
                converted.cindex.contour,
                converted.cindex.point,
            ),
            (0, 0, -1)
        );
        assert_ne!(converted.obj[0].flags & IMOD_OBJFLAG_OPEN, 0);
        assert_ne!(converted.obj[0].cont[0].flags & ICONT_WILD, 0);
        // wmod2imod.c obtains each object with imodNewObject, not a zeroed Iobj.
        assert_eq!(
            converted.obj[0].flags & ((1 << 27) | (1 << 28)),
            (1 << 27) | (1 << 28)
        );
        assert_eq!(converted.obj[0].drawmode, 1);
        assert_eq!(converted.obj[0].symbol, 1);
        assert_eq!(converted.obj[0].symsize, 3);
        assert_eq!(converted.obj[0].linewidth, 1);
        imod_file_write(&converted, &imod).unwrap();
        let reread = imod_read(&imod).unwrap();
        assert_eq!(reread.obj[0].cont[0].pts, source.obj[0].cont[0].pts);
        std::fs::remove_file(wimp).unwrap();
        std::fs::remove_file(imod).unwrap();
    }
}
