//! Translation of `IMOD/libcfshr/colormap.c`.
#![allow(dead_code)]

use std::io::BufRead;

use crate::imod::libcfshr::b3dutil::b3d_error;

static STANDARD_RAMP_DATA: [i32; 57] = [
    14, 100, 75, 200, -530, 60, 96, 255, -469, 0, 175, 177, -400, 0, 191, 143, -383, 0, 207, 78,
    -361, 90, 255, 60, -305, 191, 255, 0, -259, 239, 255, 0, -240, 255, 255, 0, -229, 255, 175, 0,
    -172, 255, 105, 0, -122, 255, 45, 55, -60, 255, 0, 90, -40, 255, 0, 192, 10,
];
static INVERTED_RAMP_DATA: [i32; 57] = [
    14, 255, 0, 192, -10, 255, 0, 90, 40, 255, 45, 55, 60, 255, 105, 0, 122, 255, 175, 0, 172, 255,
    255, 0, 229, 239, 255, 0, 240, 191, 255, 0, 259, 90, 255, 60, 305, 0, 207, 78, 361, 0, 191,
    143, 383, 0, 175, 177, 400, 60, 96, 255, 469, 100, 75, 200, 530,
];

unsafe extern "C" {
    static mut stderr: *mut libc::FILE;
}

/// Original `cmapStandardRamp` (`colormap.c:58`).
pub fn cmap_standard_ramp() -> &'static [i32; 57] {
    &STANDARD_RAMP_DATA
}

/// Original `cmapInvertedRamp` (`colormap.c:66`).
pub fn cmap_inverted_ramp() -> &'static [i32; 57] {
    &INVERTED_RAMP_DATA
}

/// Original `cmapConvertRamp` (`colormap.c:77`).
pub fn cmap_convert_ramp(ramp_data: &[i32], table: &mut [[u8; 256]; 3]) -> i32 {
    let nline = ramp_data[0] as usize;
    let mut inramp = Vec::new();
    if inramp.try_reserve_exact(nline * 4).is_err() {
        return 1;
    }
    for i in 0..nline * 4 {
        inramp.push(ramp_data[i + 1]);
    }
    let tabscl = (inramp[nline * 4 - 1] - inramp[3]) as f32 / 255.0;
    let mut indtab = 0_usize;
    for i in 0..256 {
        let tabpos = i as f32 * tabscl + inramp[3] as f32;
        if tabpos > inramp[(indtab + 1) * 4 + 3] as f32 {
            indtab += 1;
            if indtab > nline - 2 {
                indtab -= 1;
            }
        }
        let terpfc = (tabpos - inramp[indtab * 4 + 3] as f32)
            / (inramp[(indtab + 1) * 4 + 3] - inramp[indtab * 4 + 3]) as f32;
        table[0][i] = ((1. - terpfc) * inramp[indtab * 4] as f32
            + terpfc * inramp[(indtab + 1) * 4] as f32) as u8;
        table[1][i] = ((1. - terpfc) * inramp[indtab * 4 + 1] as f32
            + terpfc * inramp[(indtab + 1) * 4 + 1] as f32) as u8;
        table[2][i] = ((1. - terpfc) * inramp[indtab * 4 + 2] as f32
            + terpfc * inramp[(indtab + 1) * 4 + 2] as f32) as u8;
    }
    0
}

/// Original `cmapReadConvert` (`colormap.c:138`).
pub fn cmap_read_convert(filename: &str, table: &mut [[u8; 256]; 3]) -> i32 {
    let file = match std::fs::File::open(filename) {
        Ok(file) => file,
        Err(_) => {
            unsafe {
                b3d_error(
                    stderr,
                    format_args!("cmapReadConvert: error opening file {}\n", filename),
                );
            }
            return 1;
        }
    };
    let mut reader = std::io::BufReader::new(file);
    let mut line = String::new();
    let mut error = 0;
    let nlines: usize;
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        error = 2;
        nlines = 0;
    } else {
        nlines = line.trim().parse::<i32>().unwrap_or(0).max(0) as usize;
        if nlines == 0 || nlines > 256 {
            unsafe {
                b3d_error(
                    stderr,
                    format_args!(
                        "cmapReadConvert: invalid number of lines ({}) in {}\n",
                        nlines, filename
                    ),
                );
            }
            error = 3;
        }
    }
    let mut ramp_data = Vec::new();
    if error == 0 && nlines < 256 {
        if ramp_data.try_reserve_exact(1 + nlines * 4).is_err() {
            error = 4;
        } else {
            ramp_data.push(nlines as i32);
        }
    }
    for i in 0..nlines {
        if error != 0 {
            break;
        }
        line.clear();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            error = 2;
        } else {
            let mut values = [0_i32; 4];
            let mut number = 0;
            let mut sign = 1;
            let mut present = false;
            let mut which = 0;
            for byte in line.bytes() {
                if byte == b'-' && !present {
                    sign = -1;
                    present = true;
                } else if byte.is_ascii_digit() {
                    number = number * 10 + (byte - b'0') as i32;
                    present = true;
                } else if present {
                    if which < 4 {
                        values[which] = sign * number;
                        which += 1;
                    }
                    number = 0;
                    sign = 1;
                    present = false;
                }
            }
            if present && which < 4 {
                values[which] = sign * number;
            }
            if nlines < 256 {
                ramp_data.extend_from_slice(&values);
            } else {
                table[0][i] = values[0] as u8;
                table[1][i] = values[1] as u8;
                table[2][i] = values[2] as u8;
            }
        }
    }
    if error == 0 && nlines < 256 {
        error = cmap_convert_ramp(&ramp_data, table);
        if error != 0 {
            error = 4;
        }
    }
    if error == 2 {
        unsafe {
            b3d_error(
                stderr,
                format_args!("cmapReadConvert: error reading file {}\n", filename),
            );
        }
    } else if error == 4 {
        unsafe {
            b3d_error(
                stderr,
                format_args!("cmapReadConvert: memory allocation error"),
            )
        }
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_standard_and_inverted_ramps_convert() {
        let mut standard = [[0; 256]; 3];
        let mut inverted = [[0; 256]; 3];
        assert_eq!(cmap_convert_ramp(cmap_standard_ramp(), &mut standard), 0);
        assert_eq!(cmap_convert_ramp(cmap_inverted_ramp(), &mut inverted), 0);
        assert_eq!(standard[0][0], 100);
        assert_eq!(standard[2][255], 192);
        assert_eq!(inverted[0][0], 255);
        assert_eq!(inverted[2][255], 200);
    }

    #[test]
    fn source_direct_256_entry_file() {
        let path = std::env::temp_dir().join(format!("imod-rs-cmap-{}", std::process::id()));
        let mut content = String::from("256\n");
        for i in 0..256 {
            content.push_str(&format!("{} {} {}\n", i, 255 - i, i / 2));
        }
        std::fs::write(&path, content).unwrap();
        let mut table = [[0; 256]; 3];
        assert_eq!(cmap_read_convert(path.to_str().unwrap(), &mut table), 0);
        assert_eq!(table[0][17], 17);
        assert_eq!(table[1][17], 238);
        assert_eq!(table[2][17], 8);
        std::fs::remove_file(path).unwrap();
    }
}
