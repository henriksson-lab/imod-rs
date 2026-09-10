//! ASCII WIMP model reader from `IMOD/flib/subrs/model/read_mod.f`.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::imod::libimod::imodel::{Icont, Imod, Iobj, Ipoint};

/// Original: `read_mod` (`read_mod.f:18`).
pub fn read_mod(path: impl AsRef<Path>) -> Result<Imod, ()> {
    let file = File::open(path).map_err(|_| ())?;
    let mut lines = BufReader::new(file).lines();
    let first = lines.next().ok_or(())?.map_err(|_| ())?;
    if !(first.starts_with("Model") || first.starts_with(" Model")) {
        return Err(());
    }
    let indentation = if first.starts_with("Model") { 0 } else { 1 };
    // The three count lines and the object-sequence heading are consumed by
    // the source before it begins its Object # loop.
    for _ in 0..4 {
        lines.next().ok_or(())?.map_err(|_| ())?;
    }
    let mut imod = Imod::default();
    let mut current = lines.next().ok_or(())?.map_err(|_| ())?;
    loop {
        let field = current.get(indentation..).unwrap_or("");
        if field.starts_with(" Obj") {
            let object_number: usize = field
                .strip_prefix(" Object #:")
                .or_else(|| field.strip_prefix("Object #:"))
                .ok_or(())?
                .trim()
                .parse()
                .map_err(|_| ())?;
            let point_line = lines.next().ok_or(())?.map_err(|_| ())?;
            let point_field = point_line.get(indentation..).unwrap_or("");
            let points: usize = point_field
                .strip_prefix(" # of point:")
                .or_else(|| point_field.strip_prefix("# of point:"))
                .ok_or(())?
                .trim()
                .parse()
                .map_err(|_| ())?;
            let display_line = lines.next().ok_or(())?.map_err(|_| ())?;
            let display_field = display_line.get(indentation..).unwrap_or("");
            let display: i32 = display_field
                .split(':')
                .nth(1)
                .ok_or(())?
                .split_whitespace()
                .last()
                .ok_or(())?
                .parse()
                .map_err(|_| ())?;
            lines.next().ok_or(())?.map_err(|_| ())?;
            let mut contour = Icont::default();
            for _ in 0..points {
                let point_line = lines.next().ok_or(())?.map_err(|_| ())?;
                let values: Vec<&str> = point_line.split_whitespace().collect();
                if values.len() < 5 {
                    return Err(());
                }
                contour.pts.push(Ipoint {
                    x: values[1].parse().map_err(|_| ())?,
                    y: values[2].parse().map_err(|_| ())?,
                    z: values[3].parse().map_err(|_| ())?,
                });
            }
            // `read_mod` keeps contour entries by WIMP object number.  The
            // direct Imod representation is one contour per object, which is
            // equivalent for convertmod's following `imod_to_wmod` call.
            while imod.obj.len() < object_number {
                imod.obj.push(Iobj::default());
            }
            imod.obj[object_number - 1] = Iobj {
                name: format!("Wimp no. {display}"),
                flags: crate::imod::libimod::imodel::IMOD_OBJFLAG_OPEN,
                cont: vec![contour],
                ..Iobj::default()
            };
            current = lines.next().ok_or(())?.map_err(|_| ())?;
        // Formatted Fortran reads blank records into a blank-padded CHAR*80;
        // Rust's line reader returns an empty String for that same record.
        } else if current.trim().is_empty() {
            return Ok(imod);
        } else {
            return Err(());
        }
    }
}
