//! Translation of `IMOD/mrc/mrclog.c`.

use crate::imod::libcfshr::b3dutil::{ImodFile, SEEK_SET, b3d_fread, b3d_fseek, b3d_fwrite};
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT, MRC_MODE_SHORT,
    MrcHeader, mrc_head_label, mrc_head_read, mrc_head_write,
};

/// C `main` in `mrclog.c`, returning its process status for the launcher.
pub fn mrclog(arguments: &[String]) -> i32 {
    let program = arguments.first().map_or("mrclog", String::as_str);
    if arguments.len() < 3 {
        eprintln!("Usage: {program} [infile] [outfile]");
        return 3;
    }
    let Some(mut input) = ImodFile::open(&arguments[1], "rb") else {
        eprintln!("Error opening {}.", arguments[1]);
        return 3;
    };
    let Some(mut output) = ImodFile::open(&arguments[2], "wb") else {
        eprintln!("Error opening {}.", arguments[2]);
        return 3;
    };
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut header) != 0 {
        eprintln!("Can't Read Input File Header.");
        return 3;
    }
    let Some(mut data_size) = usize::try_from(header.nx).ok().and_then(|x| {
        usize::try_from(header.ny)
            .ok()
            .and_then(|y| x.checked_mul(y))
            .and_then(|xy| {
                usize::try_from(header.nz)
                    .ok()
                    .and_then(|z| xy.checked_mul(z))
            })
    }) else {
        eprintln!("{program}: invalid image dimensions.");
        return 3;
    };
    if matches!(header.mode, MRC_MODE_COMPLEX_SHORT | MRC_MODE_COMPLEX_FLOAT) {
        let Some(size) = data_size.checked_mul(2) else {
            eprintln!("{program}: image is too large.");
            return 3;
        };
        data_size = size;
    }
    mrc_head_label(&mut header, b"mrclog: Took log base 10 of data");
    if mrc_head_write(&mut output, &mut header) != 0 {
        return 3;
    }

    match header.mode {
        MRC_MODE_BYTE => {
            for _ in 0..data_size {
                let mut pixel = [0_u8; 1];
                if b3d_fread(&mut pixel, 1, 1, &mut input) != 1 {
                    return 3;
                }
                pixel[0] = (pixel[0] as f32).ln() as u8;
                if b3d_fwrite(&pixel, 1, 1, &mut output) != 1 {
                    return 3;
                }
            }
        }
        MRC_MODE_SHORT | MRC_MODE_COMPLEX_SHORT => {
            for _ in 0..data_size {
                let mut bytes = [0_u8; 2];
                if b3d_fread(&mut bytes, 2, 1, &mut input) != 1 {
                    return 3;
                }
                bytes = ((i16::from_ne_bytes(bytes) as f32).ln() as i16).to_ne_bytes();
                if b3d_fwrite(&bytes, 2, 1, &mut output) != 1 {
                    return 3;
                }
            }
        }
        MRC_MODE_FLOAT | MRC_MODE_COMPLEX_FLOAT => {
            let mut bytes = [0_u8; 4];
            if b3d_fseek(&mut input, 1024, SEEK_SET) != 0 {
                return 3;
            }
            if data_size == 0 || b3d_fread(&mut bytes, 4, 1, &mut input) != 1 {
                return 3;
            }
            let mut minimum_input = f32::from_ne_bytes(bytes);
            for _ in 1..data_size {
                if b3d_fread(&mut bytes, 4, 1, &mut input) != 1 {
                    return 3;
                }
                minimum_input = minimum_input.min(f32::from_ne_bytes(bytes));
            }
            if b3d_fseek(&mut input, 1024, SEEK_SET) != 0 {
                return 3;
            }
            let add = if minimum_input < 1.0 {
                1.0 - minimum_input
            } else {
                0.0
            };
            let mut minimum = 999_999.0_f32;
            let mut maximum = 0.0_f32;
            let mut total = 0.0_f32;
            for _ in 0..data_size {
                if b3d_fread(&mut bytes, 4, 1, &mut input) != 1 {
                    return 3;
                }
                let pixel = (f32::from_ne_bytes(bytes) + add).max(1.0).ln();
                minimum = minimum.min(pixel);
                maximum = maximum.max(pixel);
                total += pixel;
                if b3d_fwrite(&pixel.to_ne_bytes(), 4, 1, &mut output) != 1 {
                    return 3;
                }
            }
            header.amin = minimum;
            header.amax = maximum;
            header.amean = total / data_size as f32;
            if mrc_head_write(&mut output, &mut header) != 0 {
                return 3;
            }
        }
        _ => {
            eprintln!("{program}: data type {} unsupported.", header.mode);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::{mrc_head_new, mrc_read_slice};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn transforms_float_stack_and_updates_the_header_statistics() {
        let number = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("imod-rs-mrclog-{}-{number}", std::process::id()));
        let input_path = root.with_extension("input.mrc");
        let output_path = root.with_extension("output.mrc");
        let mut input = ImodFile::open(&input_path, "wb").unwrap();
        let mut header = MrcHeader::default();
        assert_eq!(mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_FLOAT), 0);
        assert_eq!(mrc_head_write(&mut input, &mut header), 0);
        let mut data = Vec::new();
        data.extend_from_slice(&(-1.0_f32).to_ne_bytes());
        data.extend_from_slice(&(3.0_f32).to_ne_bytes());
        assert_eq!(b3d_fwrite(&data, 1, data.len(), &mut input), data.len());
        drop(input);
        assert_eq!(
            mrclog(&[
                "mrclog".into(),
                input_path.display().to_string(),
                output_path.display().to_string()
            ]),
            0
        );
        let mut output = ImodFile::open(&output_path, "rb").unwrap();
        let mut result = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut output, &mut result), 0);
        let mut got = vec![0_u8; 8];
        assert_eq!(
            mrc_read_slice(&mut got, &mut output, &mut result, 0, b'Z'),
            0
        );
        let values = got
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>();
        assert_eq!(values[0], 0.0);
        assert!((values[1] - 5.0_f32.ln()).abs() < 0.000001, "{values:?}");
        assert_eq!(result.amin, values[0]);
        assert_eq!(result.amax, values[1]);
        drop(output);
        let _ = std::fs::remove_file(input_path);
        let _ = std::fs::remove_file(output_path);
    }
}
