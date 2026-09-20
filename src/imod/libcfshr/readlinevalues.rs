//! Translation of `IMOD/libcfshr/readlinevalues.c`.

use std::io::BufRead;

pub const RLFV_SEPARATE_LINES: i32 = 1;

/// Rust representation of a source `readLinesForValues` variadic destination.
pub enum ReadValueArray<'a> {
    Integers(&'a mut [i32]),
    Floats(&'a mut [f32]),
    Doubles(&'a mut [f64]),
}

/// Matches `readLinesForValues` (`IMOD/libcfshr/readlinevalues.c:35`).
pub fn read_lines_for_values<R: BufRead>(
    reader: &mut R,
    number_to_get: &mut i32,
    value_size: usize,
    flags: i32,
    types: &str,
    destinations: &mut [ReadValueArray<'_>],
) -> i32 {
    if types.is_empty() || types.len() != destinations.len() {
        return -7;
    }
    let wanted = *number_to_get;
    let limit = if wanted <= 0 {
        value_size
    } else {
        wanted as usize
    };
    let separate = flags & RLFV_SEPARATE_LINES != 0;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut line = String::new();
    while rows.len() < limit {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Err(_) => return -1,
            Ok(_) => {}
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        let values = text
            .split(|character: char| {
                character == ',' || character == '/' || character.is_ascii_whitespace()
            })
            .filter(|value| !value.is_empty())
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>();
        let Ok(values) = values else {
            return -4;
        };
        if separate {
            if values.len() < types.len() {
                return -4;
            }
            rows.push(values[..types.len()].to_vec());
        } else {
            for chunk in values.chunks(types.len()) {
                if chunk.len() < types.len() {
                    break;
                }
                rows.push(chunk.to_vec());
                if rows.len() == limit {
                    break;
                }
            }
        }
    }
    if wanted > 0 && rows.len() < limit {
        return -2;
    }
    for (column, (kind, destination)) in types.bytes().zip(destinations.iter_mut()).enumerate() {
        match (kind, destination) {
            (b'i', ReadValueArray::Integers(output)) => {
                for (row, values) in rows.iter().enumerate() {
                    if row >= output.len() {
                        return -3;
                    }
                    let integer = values[column].round() as i32;
                    if (values[column] - integer as f64).abs() > 0.001 {
                        return -5;
                    }
                    output[row] = integer;
                }
            }
            (b'f', ReadValueArray::Floats(output)) => {
                for (row, values) in rows.iter().enumerate() {
                    if row >= output.len() {
                        return -3;
                    }
                    output[row] = values[column] as f32;
                }
            }
            (b'd', ReadValueArray::Doubles(output)) => {
                for (row, values) in rows.iter().enumerate() {
                    if row >= output.len() {
                        return -3;
                    }
                    output[row] = values[column];
                }
            }
            _ => return -7,
        }
    }
    if wanted <= 0 {
        *number_to_get = rows.len() as i32;
    }
    0
}

/// Matches `exitFromValueReadError` (`IMOD/libcfshr/readlinevalues.c:221`).
pub fn exit_from_value_read_error(error: i32, description: &str) -> Result<(), String> {
    match error {
        0 => Ok(()),
        -1 => Err(format!("Reading a line from file of {description}")),
        -2 => Err(format!(
            "End of file before all values gotten from file of {description}"
        )),
        -3 => Err(format!(
            "Array full before all values gotten from file of {description}"
        )),
        -4 => Err(format!("Parsing values on line from file of {description}")),
        -5 => Err(format!(
            "Non-integer value when expecting an integer in file of {description}"
        )),
        -6 => Err(format!(
            "Allocating temporary array to get values from file of {description}"
        )),
        -7 => Err(format!(
            "Programming error trying to get values from file of {description}"
        )),
        _ => Err(format!(
            "Unknown error code ({error}) reading values from file of {description}"
        )),
    }
}
