//! `IMOD/imodutil/wmod2imod.c`: convert a legacy ASCII WIMP model to IMOD.

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

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
                    xscale = argv
                        .get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(xscale)
                }
                Some(b'y') => {
                    i += 1;
                    yscale = argv
                        .get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(yscale)
                }
                Some(b'z') => {
                    i += 1;
                    zscale = argv
                        .get(i)
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(zscale)
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
    let input = &argv[i];
    let mut fin = File::open(input).unwrap_or_else(|_| {
        // C increments i before this diagnostic, so it prints the output name.
        eprintln!("Couldn't open {}", argv.get(i + 1).unwrap_or(input));
        std::process::exit(3)
    });
    i += 1;
    let output = &argv[i];
    let mut fout = File::create(output).unwrap_or_else(|_| {
        eprintln!("Couldn't open {output}");
        std::process::exit(3)
    });
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
pub fn imod_from_wmod(fin: &mut File) -> Option<Imod> {
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
    let mut text = String::new();
    fin.seek(SeekFrom::Start(0)).ok()?;
    fin.read_to_string(&mut text).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    let mut objlookup = [-1_i32; 256];
    let mut used = [false; 256];
    let mut line_index = 0_usize;
    // First source pass: make the WIMP display-switch-to-object lookup.
    while line_index < lines.len() {
        let line = lines[line_index];
        let Some(offset) = line.find('O') else {
            line_index += 1;
            continue;
        };
        if !line[offset..].starts_with("Object #:") {
            line_index += 1;
            continue;
        }
        let points_line = *lines.get(line_index + 1)?;
        let switch_line = *lines.get(line_index + 2)?;
        let _points = points_line
            .split_whitespace()
            .nth(3)?
            .parse::<usize>()
            .ok()?;
        let display_switch = switch_line.split_whitespace().nth(2)?.parse::<i32>().ok()?;
        if (0..256).contains(&display_switch) {
            used[display_switch as usize] = true;
        }
        line_index += 3;
    }
    let mut model = Imod::default();
    // `wmod2imod.c:171` obtains this through `imodNew`, whose `imodDefault`
    // state is written into the resulting binary model.
    model.name = "IMOD-NewModel".to_owned();
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
    for wimp_number in 0..256_usize {
        if !used[wimp_number] {
            continue;
        }
        objlookup[wimp_number] = model.obj.len() as i32;
        let (red, green, blue) = if wimp_number >= 247 {
            let color = WMOD_COLORS[wimp_number - 247];
            (color[0], color[1], color[2])
        } else {
            let color = wimp_number as f32 / 255.0;
            (color, color, color)
        };
        if imod_new_object(&mut model) != 0 {
            return None;
        }
        let object = model.obj.last_mut()?;
        object.name = format!("Wimp no. {wimp_number}");
        object.red = red;
        object.green = green;
        object.blue = blue;
    }
    line_index = 0;
    // Second source pass: turn each WIMP object record into one IMOD contour.
    while line_index < lines.len() {
        let line = lines[line_index];
        let Some(offset) = line.find('O') else {
            line_index += 1;
            continue;
        };
        if !line[offset..].starts_with("Object #:") {
            line_index += 1;
            continue;
        }
        let points = lines
            .get(line_index + 1)?
            .split_whitespace()
            .nth(3)?
            .parse::<usize>()
            .ok()?;
        let display_switch = lines
            .get(line_index + 2)?
            .split_whitespace()
            .nth(2)?
            .parse::<i32>()
            .ok()?;
        let object_index =
            if (0..256).contains(&display_switch) && objlookup[display_switch as usize] >= 0 {
                objlookup[display_switch as usize] as usize
            } else {
                0
            };
        // `wmod2imod.c` sets the current contour to the previous one and
        // creates the next contour with `imodNewContour`; that routine updates
        // the persisted current index and inherits its source contour state.
        model.cindex.object = object_index as i32;
        model.cindex.contour = model.obj.get(object_index)?.cont.len() as i32 - 1;
        if imod_new_contour(&mut model) != 0 {
            return None;
        }
        let contour_index = model.cindex.contour as usize;
        let contour = model
            .obj
            .get_mut(object_index)?
            .cont
            .get_mut(contour_index)?;
        for point_index in 0..points {
            let point_line = *lines.get(line_index + 4 + point_index)?;
            let mut values = point_line.split_whitespace();
            let _number = values.next()?;
            if imod_point_add(
                contour,
                Some(Ipoint {
                    x: values.next()?.parse().ok()?,
                    y: values.next()?.parse().ok()?,
                    z: values.next()?.parse().ok()?,
                }),
                point_index as i32,
            ) == 0
            {
                return None;
            }
        }
        let object = model.obj.get_mut(object_index)?;
        if object.cont[0].pts.len() < 2 || object.cont[0].pts[0].z != object.cont[0].pts[1].z {
            object.flags |= IMOD_OBJFLAG_OPEN;
        }
        line_index += 4 + points;
    }
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
        let mut wimp_file = File::open(&wimp).unwrap();
        let converted = imod_from_wmod(&mut wimp_file).unwrap();
        assert_eq!(converted.obj.len(), 1);
        assert_eq!(converted.obj[0].name, "Wimp no. 247");
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
