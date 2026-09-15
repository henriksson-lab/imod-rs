//! Translation of `IMOD/mrc/mrctilt.c`.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{MrcHeader, mrc_head_read, mrc_head_write};

/// Apply the source's tilt-header conversion to an already loaded MRC header.
/// `vd1` and `vd2` are stored in hundredths of a degree by this legacy format.
pub fn apply_tilt_information(header: &mut MrcHeader, first_tilt: f32, tilt_increment: f32) {
    header.idtype = 1;
    header.vd1 = (first_tilt * 100.0) as i16;
    header.vd2 = (tilt_increment * 100.0) as i16;
}

/// C `main` in `mrctilt.c`, returning its process status for the launcher.
pub fn mrctilt(arguments: &[String]) -> i32 {
    let program = arguments.first().map_or("mrctilt", String::as_str);
    if arguments.len() != 4 {
        eprintln!("{program} version 1.0 Copyright (C)1994 Boulder Laboratory for");
        eprintln!("3-Dimensional Fine Structure, University of Colorado.");
        eprintln!("{program}: Modify a mrc header to contain tilt information.");
        eprintln!("Usage: {program} [image file] [first tilt] [tilt increment]");
        return 3;
    }
    let filename = &arguments[1];
    let first_tilt = match arguments[2].parse::<f32>() {
        Ok(value) => value,
        Err(_) => {
            eprintln!("Invalid first tilt {}.", arguments[2]);
            return 3;
        }
    };
    let tilt_increment = match arguments[3].parse::<f32>() {
        Ok(value) => value,
        Err(_) => {
            eprintln!("Invalid tilt increment {}.", arguments[3]);
            return 3;
        }
    };
    let Some(mut input) = ImodFile::open(filename, "rb") else {
        eprintln!("Error opening {filename}.");
        return 3;
    };
    let Some(mut output) = ImodFile::open(filename, "rb+") else {
        // The C source's typo tests `fin` here.  Report the intended output
        // failure instead of silently attempting a write through no handle.
        eprintln!("Error opening {filename}.");
        return 3;
    };
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut header) != 0 {
        eprintln!("Can't Read Input File Header.");
        return 3;
    }
    println!("First tilt = {first_tilt}, Inc = {tilt_increment}");
    apply_tilt_information(&mut header, first_tilt, tilt_increment);
    if mrc_head_write(&mut output, &mut header) != 0 {
        eprintln!("Can't Write Output File Header.");
        return 3;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{apply_tilt_information, mrctilt};
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FILE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn stores_tilts_in_hundredths_of_a_degree() {
        let mut header = MrcHeader::default();
        apply_tilt_information(&mut header, -61.25, 1.5);
        assert_eq!((header.idtype, header.vd1, header.vd2), (1, -6125, 150));
    }

    #[test]
    fn updates_a_real_mrc_header_in_place() {
        let sequence = FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "imod-rs-mrctilt-{}-{sequence}.mrc",
            std::process::id()
        ));
        let mut output = ImodFile::open(&path, "wb").expect("create MRC");
        let mut header = MrcHeader::default();
        mrc_head_new(&mut header, 2, 2, 1, MRC_MODE_BYTE);
        assert_eq!(mrc_head_write(&mut output, &mut header), 0);
        drop(output);
        assert_eq!(
            mrctilt(&[
                "mrctilt".into(),
                path.display().to_string(),
                "-2.5".into(),
                "0.75".into()
            ]),
            0
        );
        let mut input = ImodFile::open(&path, "rb").expect("reopen MRC");
        let mut updated = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut input, &mut updated), 0);
        assert_eq!((updated.idtype, updated.vd1, updated.vd2), (1, -250, 75));
        let _ = std::fs::remove_file(path);
    }
}
