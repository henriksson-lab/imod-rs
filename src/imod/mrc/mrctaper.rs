//! Translation of `IMOD/mrc/mrctaper.c`.
//!
//! The command uses an owned [`Islice`] rather than the source's malloc buffer;
//! MRC handles remain only at the read/write boundary.

use std::io::Write;

use crate::imod::libcfshr::b3dutil::{ImodFile, b3d_output_file_type, imod_prog_name};
use crate::imod::libcfshr::islice::{slice_create, slice_mode_if_real};
use crate::imod::libcfshr::taperatfill::slice_taper_at_fill;
use crate::imod::libiimod::iimage::IIFILE_TIFF;
use crate::imod::libiimod::mrcfiles::{
    MrcHeader, mrc_head_label, mrc_head_read, mrc_head_write, mrc_init_output_header,
    mrc_read_slice, mrc_write_slice,
};

const DEFAULT_TAPER: i32 = 16;

/// `mrctaper_help` (`mrctaper.c:23`).
pub fn mrctaper_help(name: &str) {
    eprintln!("Usage: {name} [-i] [-t #] [-z min,max] input_file [output_file]");
    println!("Options:");
    println!("\t-i\tTaper inside (default is outside).");
    println!("\t-t #\tTaper over the given # of pixels (default {DEFAULT_TAPER} or 1% of size).");
    println!("\t-z min,max\tDo only sections between min and max.");
    println!("\tWith no output file, images are written back to input file.");
}

fn parse_range(value: &str) -> Option<(i32, i32)> {
    let (first, second) = value.split_once(',').or_else(|| value.split_once(':'))?;
    Some((first.trim().parse().ok()?, second.trim().parse().ok()?))
}

/// C `main` in `mrctaper.c`, returning its process status for the dispatcher.
pub fn mrctaper(arguments: &[String]) -> i32 {
    let progname = arguments
        .first()
        .map(|argument| imod_prog_name(argument))
        .unwrap_or_else(|| "mrctaper".to_owned());
    if arguments.len() < 2 {
        mrctaper_help(&progname);
        return 3;
    }

    let (mut inside, mut ntaper, mut taper_entered, mut zmin, mut zmax) =
        (false, DEFAULT_TAPER, false, -1, -1);
    let mut index = 1;
    while let Some(argument) = arguments.get(index) {
        if !argument.starts_with('-') || argument == "-" {
            break;
        }
        match argument.as_bytes().get(1).copied() {
            Some(b'i') if argument.len() == 2 => inside = true,
            Some(b't') => {
                taper_entered = true;
                let value = if argument.len() > 2 {
                    &argument[2..]
                } else {
                    index += 1;
                    let Some(value) = arguments.get(index) else {
                        eprintln!("ERROR: {progname} - missing taper width");
                        return 1;
                    };
                    value
                };
                let Ok(value) = value.parse() else {
                    eprintln!("ERROR: {progname} - invalid taper width");
                    return 1;
                };
                ntaper = value;
            }
            Some(b'z') => {
                let value = if argument.len() > 2 {
                    &argument[2..]
                } else {
                    index += 1;
                    let Some(value) = arguments.get(index) else {
                        eprintln!("ERROR: {progname} - missing section range");
                        return 1;
                    };
                    value
                };
                let Some((minimum, maximum)) = parse_range(value) else {
                    eprintln!("ERROR: {progname} - invalid section range");
                    return 1;
                };
                zmin = minimum;
                zmax = maximum;
            }
            _ => {
                println!("ERROR: {progname} - illegal option");
                mrctaper_help(&progname);
                return 1;
            }
        }
        index += 1;
    }
    let positional = &arguments[index..];
    if positional.is_empty() || positional.len() > 2 {
        mrctaper_help(&progname);
        return 3;
    }
    if !(1..=127).contains(&ntaper) {
        eprintln!("ERROR: {progname} - Taper must be between 1 and 127.");
        return 1;
    }

    let input_name = &positional[0];
    let writing_input = positional.len() == 1;
    let Some(mut input) = ImodFile::open(input_name, if writing_input { "rb+" } else { "rb" })
    else {
        eprintln!("ERROR: {progname} - Opening {input_name}.");
        return 1;
    };
    let mut input_header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut input_header) != 0 {
        eprintln!("ERROR: {progname} - Can't Read Input File Header.");
        return 1;
    }
    if slice_mode_if_real(input_header.mode) < 0 {
        eprintln!("ERROR: {progname} - Can operate only on byte, integer and real data.");
        return 1;
    }
    if !taper_entered {
        ntaper = ((input_header.nx + input_header.ny) / 200).clamp(DEFAULT_TAPER, 127);
        println!("Tapering over {ntaper} pixels");
    }
    if zmin == -1 && zmax == -1 {
        zmin = 0;
        zmax = input_header.nz - 1;
    } else {
        zmin = zmin.max(0);
        zmax = zmax.min(input_header.nz - 1);
    }

    let (mut output, mut output_header, section_offset) =
        if let Some(output_name) = positional.get(1) {
            let Some(output) = ImodFile::open(output_name, "wb") else {
                eprintln!("ERROR: {progname} - Opening {output_name}.");
                return 1;
            };
            let mut header = input_header.clone();
            mrc_init_output_header(&mut header);
            header.nz = zmax + 1 - zmin;
            header.mz = header.nz;
            header.zlen = header.nz as f32;
            (output, header, zmin)
        } else {
            if b3d_output_file_type() == IIFILE_TIFF {
                eprintln!("ERROR: {progname} - Cannot write to an existing TIFF file.");
                return 1;
            }
            (input.clone(), input_header.clone(), 0)
        };
    let Some(mut slice) = slice_create(input_header.nx, input_header.ny, input_header.mode) else {
        eprintln!("ERROR: {progname} - Couldn't get memory for slice.");
        return 1;
    };
    for section in zmin..=zmax {
        print!("\rDoing section #{section:4}");
        let _ = std::io::stdout().flush();
        if mrc_read_slice(
            slice.data.bytes_mut(),
            &mut input,
            &mut input_header,
            section,
            b'Z',
        ) != 0
        {
            eprintln!("\nERROR: {progname} - Reading section {section}.");
            return 1;
        }
        if slice_taper_at_fill(&mut slice, ntaper, inside) != 0 {
            eprintln!("\nERROR: {progname} - Can't get memory for taper operation.");
            return 1;
        }
        if mrc_write_slice(
            slice.data.bytes(),
            &mut output,
            &mut output_header,
            section - section_offset,
            b'Z',
        ) != 0
        {
            eprintln!("\nERROR: {progname} - Writing section {section}.");
            return 1;
        }
    }
    println!("\nDone!");
    mrc_head_label(
        &mut output_header,
        b"mrctaper: Image tapered down to fill value at edges",
    );
    if mrc_head_write(&mut output, &mut output_header) != 0 {
        eprintln!("ERROR: {progname} - Writing output header.");
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::mrctaper;
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_BYTE, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write, mrc_read_slice,
        mrc_write_slice,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FILE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn tapers_a_real_mrc_into_a_new_output_stack() {
        let number = FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("imod-rs-mrctaper-{}-{number}", std::process::id()));
        let input_path = root.with_extension("input.mrc");
        let output_path = root.with_extension("output.mrc");
        let mut input = ImodFile::open(&input_path, "wb").expect("create input");
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 14, 14, 2, MRC_MODE_BYTE), 0);
        let plane = (0..14)
            .flat_map(|y| {
                (0..14).map(move |x| {
                    if x == 0 || x == 13 || y == 0 || y == 13 {
                        0
                    } else {
                        100
                    }
                })
            })
            .collect::<Vec<u8>>();
        for section in 0..2 {
            assert_eq!(
                mrc_write_slice(&plane, &mut input, &mut header, section, b'Z'),
                0
            );
        }
        assert_eq!(mrc_head_write(&mut input, &mut header), 0);
        drop(input);
        assert_eq!(
            mrctaper(&[
                "mrctaper".into(),
                "-t2".into(),
                input_path.display().to_string(),
                output_path.display().to_string()
            ]),
            0
        );
        let mut output = ImodFile::open(&output_path, "rb").expect("open output");
        let mut output_header = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut output, &mut output_header), 0);
        assert_eq!(output_header.nz, 2);
        assert!(
            output_header.labels[..output_header.nlabl as usize]
                .iter()
                .any(|label| label.starts_with(b"mrctaper:"))
        );
        let mut tapered = vec![0_u8; 196];
        assert_eq!(
            mrc_read_slice(&mut tapered, &mut output, &mut output_header, 0, b'Z'),
            0
        );
        assert_ne!(tapered, plane);
        let _ = std::fs::remove_file(input_path);
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn rejects_invalid_taper_width_before_opening_input() {
        assert_eq!(
            mrctaper(&["mrctaper".into(), "-t0".into(), "missing.mrc".into()]),
            1
        );
    }
}
