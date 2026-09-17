//! Translation of `IMOD/mrc/tifinfo.c`.

use std::fmt::Write as _;

/// `swap` (`tifinfo.c:37`).  The source reverses the largest even prefix and
/// leaves a trailing odd byte untouched.
pub fn swap(bytes: &mut [u8]) {
    let even_length = bytes.len() & !1;
    bytes[..even_length].reverse();
}

/// C `tiff_print_info`, operating on owned file bytes instead of a C stream.
pub fn tiff_print_info(data: &[u8], verbose: bool) -> String {
    if data.len() < 8 {
        return "Not a tif file: file is shorter than a TIFF header\n".into();
    }
    let order = u16::from_ne_bytes([data[0], data[1]]);
    if order != 0x4949 && order != 0x4d4d {
        return format!("Not a tif file: first word must be 0x4949 or 0x4d4d,  not {order:x}\n");
    }
    let little = order == 0x4949;
    let read16 = |at: usize| -> Option<u16> {
        data.get(at..at + 2).map(|part| {
            if little {
                u16::from_le_bytes([part[0], part[1]])
            } else {
                u16::from_be_bytes([part[0], part[1]])
            }
        })
    };
    let read32 = |at: usize| -> Option<u32> {
        data.get(at..at + 4).map(|part| {
            if little {
                u32::from_le_bytes([part[0], part[1], part[2], part[3]])
            } else {
                u32::from_be_bytes([part[0], part[1], part[2], part[3]])
            }
        })
    };
    let mut report = format!("TIFF: {order:x}");
    let Some(version) = read16(2) else {
        return report;
    };
    if version != 42 {
        let _ = writeln!(
            report,
            "\nBad TIFF Version number. Must be 42, not {version}"
        );
        return report;
    }
    let _ = writeln!(report, " {version:x}");
    let mut offset = read32(4).unwrap_or(0) as usize;
    let mut record = 1;
    while offset != 0 {
        let Some(entries) = read16(offset) else {
            break;
        };
        let _ = writeln!(
            report,
            "Reading {entries} entries in record {record} at {offset}:"
        );
        record += 1;
        for entry in 0..entries as usize {
            let at = offset + 2 + entry * 12;
            let (Some(tag), Some(kind), Some(length), Some(mut value)) =
                (read16(at), read16(at + 2), read32(at + 4), read32(at + 8))
            else {
                return report;
            };
            if !little && kind == 3 && length < 3 {
                value >>= 16;
            }
            let name = match kind {
                0 => "NULL ",
                1 => "BYTE ",
                2 => "ASCII",
                3 => "SHORT",
                4 => "LONG ",
                5 => "RATIONAL",
                _ => "UNKNOWN",
            };
            let _ = write!(report, "\t{tag} {name} {length} {value} ");
            let size = match kind {
                1 | 2 => length as usize,
                3 => length as usize * 2,
                4 => length as usize * 4,
                _ => 0,
            };
            if verbose && size > 4 {
                let _ = write!(report, "\t : ");
                let start = value as usize;
                for item in 0..length as usize {
                    match kind {
                        1 => {
                            if let Some(byte) = data.get(start + item) {
                                let _ = write!(report, "{byte} ");
                            }
                        }
                        2 => {
                            if let Some(byte) = data.get(start + item) {
                                report.push(*byte as char);
                            }
                        }
                        3 => {
                            if let Some(number) = read16(start + item * 2) {
                                let _ = write!(report, "{number} ");
                            }
                        }
                        4 => {
                            if let Some(number) = read32(start + item * 4) {
                                let _ = write!(report, "{number} ");
                            }
                        }
                        _ => {}
                    }
                }
            }
            report.push('\n');
        }
        offset = read32(offset + 2 + entries as usize * 12).unwrap_or(0) as usize;
    }
    report
}

/// C `main` in `tifinfo.c`.
pub fn tifinfo(arguments: &[String]) -> i32 {
    let mut first = 1;
    let mut verbose = false;
    if arguments.get(1).is_some_and(|value| value == "-v") {
        verbose = true;
        first = 2;
    }
    for filename in &arguments[first..] {
        let Ok(data) = std::fs::read(filename) else {
            return 1;
        };
        print!("{}", tiff_print_info(&data, verbose));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::{swap, tiff_print_info, tifinfo};
    #[test]
    fn reports_a_real_little_endian_tiff_ifd() {
        let bytes = [
            b'I', b'I', 42, 0, 8, 0, 0, 0, 1, 0, 0, 1, 4, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        ];
        let report = tiff_print_info(&bytes, false);
        assert!(report.contains("TIFF: 4949 2a"));
        assert!(report.contains("Reading 1 entries"));
        assert!(report.contains("256 LONG  1 2"));
    }
    #[test]
    fn returns_failure_for_missing_file() {
        assert_eq!(tifinfo(&["tifinfo".into(), "missing.tif".into()]), 1);
    }
    #[test]
    fn source_swap_reverses_only_the_even_prefix() {
        let mut bytes = [1, 2, 3, 4, 5];
        swap(&mut bytes);
        assert_eq!(bytes, [4, 3, 2, 1, 5]);
    }
}
