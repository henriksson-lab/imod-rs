//! Translation of `IMOD/mrc/mrcx.c`.
//!
//! `mrcx` is deliberately a byte-level converter: unlike normal MRC readers it
//! must preserve the header's otherwise opaque fields while changing the byte
//! representation of numeric fields and pixels.

use std::fs;
use std::path::Path;

use crate::imod::libiimod::mrcfiles::{
    MRC_HEADER_SIZE, MRC_MODE_BYTE, MRC_MODE_COMPLEX_FLOAT, MRC_MODE_COMPLEX_SHORT, MRC_MODE_FLOAT,
    MRC_MODE_RGB, MRC_MODE_SHORT,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MrcxFloatFormat {
    /// The normal current-platform path: reverse IEEE multi-byte values.
    Ieee,
    /// The historical VAX floating-point conversion selected by `-vms`.
    Vax,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MrcxOptions {
    pub format: MrcxFloatFormat,
    pub input: String,
    pub output: Option<String>,
}

/// C `convertBytes`: copy source bytes unchanged.
pub fn convert_bytes(
    input: &[u8],
    buffer: &mut [u8],
    no_bytes: usize,
    _direction: bool,
) -> Result<(), String> {
    if input.len() < no_bytes || buffer.len() < no_bytes {
        return Err("Read Failed While Reading Byte Quantities".into());
    }
    buffer[..no_bytes].copy_from_slice(&input[..no_bytes]);
    Ok(())
}

/// C `convertShorts`: reverse every two-byte scalar.
pub fn convert_shorts(
    input: &[u8],
    buffer: &mut [u8],
    no_shorts: usize,
    _direction: bool,
) -> Result<(), String> {
    let bytes = no_shorts
        .checked_mul(2)
        .ok_or_else(|| "short conversion size overflow".to_string())?;
    if input.len() < bytes || buffer.len() < bytes {
        return Err("mrcx: Read Failed While Reading Short Quantities".into());
    }
    for (from, to) in input[..bytes]
        .chunks_exact(2)
        .zip(buffer[..bytes].chunks_exact_mut(2))
    {
        to[0] = from[1];
        to[1] = from[0];
    }
    Ok(())
}

/// C `convertLongs`: reverse every four-byte scalar.
pub fn convert_longs(
    input: &[u8],
    buffer: &mut [u8],
    no_longs: usize,
    _direction: bool,
) -> Result<(), String> {
    let bytes = no_longs
        .checked_mul(4)
        .ok_or_else(|| "long conversion size overflow".to_string())?;
    if input.len() < bytes || buffer.len() < bytes {
        return Err("mrcx: Read Failed While Reading Long Quantities".into());
    }
    for (from, to) in input[..bytes]
        .chunks_exact(4)
        .zip(buffer[..bytes].chunks_exact_mut(4))
    {
        to.copy_from_slice(&[from[3], from[2], from[1], from[0]]);
    }
    Ok(())
}

/// C `convertFloats`, including the source's VAX exponent clamping rules.
pub fn convert_floats(
    input: &[u8],
    buffer: &mut [u8],
    no_floats: usize,
    direction: bool,
) -> Result<(), String> {
    let bytes = no_floats
        .checked_mul(4)
        .ok_or_else(|| "float conversion size overflow".to_string())?;
    if input.len() < bytes || buffer.len() < bytes {
        return Err("Read Failed While Reading Float Quantities".into());
    }
    for (from, to) in input[..bytes]
        .chunks_exact(4)
        .zip(buffer[..bytes].chunks_exact_mut(4))
    {
        let mut value = [from[0], from[1], from[2], from[3]];
        let exponent = if direction {
            (value[1] << 1) | ((value[0] >> 7) & 1)
        } else {
            (value[0] << 1) | (value[1] >> 7)
        };
        if direction {
            if exponent > 3 && exponent != 0 {
                value[1] = value[1].wrapping_sub(1);
            } else if exponent <= 3 && exponent != 0 {
                value[0] = 0x80;
                value[1] &= 0x80;
                value[2] = 0;
                value[3] = 0;
            }
        } else if exponent < 253 && exponent != 0 {
            value[0] = value[0].wrapping_add(1);
        } else if exponent >= 253 {
            value[0] |= 0x7f;
            value[1] = 0xff;
            value[2] = 0xff;
            value[3] = 0xff;
        }
        value.swap(0, 1);
        if ((direction && exponent > 3) || (!direction && exponent < 253)) && exponent != 0 {
            value.swap(2, 3);
        }
        to.copy_from_slice(&value);
    }
    Ok(())
}

/// C `convertBody`, expressed over owned input/output byte buffers rather than
/// a `FILE *` and a manually allocated scan line.
pub fn convert_body(
    input: &[u8],
    nx: usize,
    ny: usize,
    nz: usize,
    size: usize,
    factor: usize,
    direction: bool,
    format: MrcxFloatFormat,
) -> Result<Vec<u8>, String> {
    let scalars = nx
        .checked_mul(ny)
        .and_then(|n| n.checked_mul(nz))
        .and_then(|n| n.checked_mul(factor))
        .ok_or_else(|| "MRC dimensions overflow".to_string())?;
    let bytes = scalars
        .checked_mul(size)
        .ok_or_else(|| "MRC data size overflow".to_string())?;
    if input.len() < bytes {
        return Err("input file ends before MRC image data".into());
    }
    let mut output = vec![0; bytes];
    match size {
        1 => convert_bytes(input, &mut output, bytes, direction)?,
        2 => convert_shorts(input, &mut output, scalars, direction)?,
        4 if format == MrcxFloatFormat::Ieee => {
            convert_longs(input, &mut output, scalars, direction)?
        }
        4 => convert_floats(input, &mut output, scalars, direction)?,
        _ => return Err("unsupported MRC scalar size".into()),
    }
    Ok(output)
}

/// C `convertHeader`.  The numeric words are exactly those swapped by the
/// source; labels and extended-header bytes retain their opaque contents.
pub fn convert_header(
    input: &[u8],
    nextra: usize,
    direction: bool,
    format: MrcxFloatFormat,
) -> Result<Vec<u8>, String> {
    if input.len() < MRC_HEADER_SIZE + nextra {
        return Err("mrcx: Error Reading header.".into());
    }
    let mut output = input[..MRC_HEADER_SIZE + nextra].to_vec();
    let swap =
        |output: &mut [u8], offset: usize, count: usize, bytes: usize| -> Result<(), String> {
            let end = offset + count * bytes;
            if end > MRC_HEADER_SIZE {
                return Err("header conversion extends beyond MRC header".into());
            }
            let source = output[offset..end].to_vec();
            match bytes {
                2 => convert_shorts(&source, &mut output[offset..end], count, direction),
                4 => convert_longs(&source, &mut output[offset..end], count, direction),
                _ => Ok(()),
            }
        };
    swap(&mut output, 0, 10, 4)?;
    swap(&mut output, 64, 3, 4)?;
    swap(&mut output, 88, 2, 4)?;
    swap(&mut output, 96, 1, 2)?;
    swap(&mut output, 128, 4, 2)?;
    swap(&mut output, 160, 6, 2)?;
    let float_offsets = [40usize, 52, 76, 136, 172];
    let float_counts = [3usize, 3, 3, 6, 6];
    for (&offset, &count) in float_offsets.iter().zip(float_counts.iter()) {
        let end = offset + count * 4;
        let source = output[offset..end].to_vec();
        if format == MrcxFloatFormat::Ieee {
            convert_longs(&source, &mut output[offset..end], count, direction)?;
        } else {
            convert_floats(&source, &mut output[offset..end], count, direction)?;
        }
    }
    // The source distinguishes the pre-MRC2000 origin/wavelength layout.
    // Its bytes overlap the later origin/cmap/rms fields, so it cannot use a
    // common swap table here.
    if &input[208..211] != b"MAP" {
        swap(&mut output, 196, 6, 2)?;
        let source = output[208..220].to_vec();
        if format == MrcxFloatFormat::Ieee {
            convert_longs(&source, &mut output[208..220], 3, direction)?;
        } else {
            convert_floats(&source, &mut output[208..220], 3, direction)?;
        }
    } else {
        for offset in [196, 200, 204, 216] {
            let source = output[offset..offset + 4].to_vec();
            if format == MrcxFloatFormat::Ieee {
                convert_longs(&source, &mut output[offset..offset + 4], 1, direction)?;
            } else {
                convert_floats(&source, &mut output[offset..offset + 4], 1, direction)?;
            }
        }
        // C flips the machine stamp only for MRC2000 headers.
        output[212] = if input[212] == 17 { 68 } else { 17 };
    }
    let source = output[220..224].to_vec();
    convert_longs(&source, &mut output[220..224], 1, direction)?;
    // C swaps 16-bit values in the extended symmetry data.
    let start = MRC_HEADER_SIZE;
    let pairs = nextra / 2;
    if pairs != 0 {
        let source = output[start..start + pairs * 2].to_vec();
        convert_shorts(
            &source,
            &mut output[start..start + pairs * 2],
            pairs,
            direction,
        )?;
    }
    Ok(output)
}

/// Parse the command line accepted on a current little-endian IMOD build.
pub fn mrcx_options(arguments: &[String]) -> Result<MrcxOptions, String> {
    if arguments.len() < 2 || arguments.len() > 3 || arguments[1].starts_with('-') {
        return Err(format!(
            "{}: Usage [filename] [opt filename]",
            arguments.first().map_or("mrcx", String::as_str)
        ));
    }
    Ok(MrcxOptions {
        format: MrcxFloatFormat::Ieee,
        input: arguments[1].clone(),
        output: arguments.get(2).cloned(),
    })
}

/// C `main`: convert an MRC stream in place or into the requested output.
pub fn mrcx(arguments: &[String]) -> Result<(), String> {
    let options = mrcx_options(arguments)?;
    let input = fs::read(&options.input)
        .map_err(|error| format!("mrcx: Couldn't open {}: {error}", options.input))?;
    if input.len() < MRC_HEADER_SIZE {
        return Err("mrcx: Error Reading header.".into());
    }
    let native = |offset: usize| {
        i32::from_ne_bytes(
            input[offset..offset + 4]
                .try_into()
                .expect("four-byte MRC word"),
        )
    };
    let swapped = |offset: usize| {
        i32::from_ne_bytes([
            input[offset + 3],
            input[offset + 2],
            input[offset + 1],
            input[offset],
        ])
    };
    let mut values = [
        native(0),
        native(4),
        native(8),
        native(12),
        native(64),
        native(68),
        native(72),
        native(92),
    ];
    let valid = |v: &[i32; 8]| {
        v[0] > 0
            && v[1] > 0
            && v[2] > 0
            && !(v[0] > 65535 && v[1] > 65535 && v[2] > 65535)
            && (0..=4).contains(&v[4])
            && (0..=4).contains(&v[5])
            && (0..=4).contains(&v[6])
            && (0..=16).contains(&v[3])
    };
    let direction = if valid(&values) {
        false
    } else {
        values = [
            swapped(0),
            swapped(4),
            swapped(8),
            swapped(12),
            swapped(64),
            swapped(68),
            swapped(72),
            swapped(92),
        ];
        if !valid(&values) {
            return Err(
                "mrcx: This is not an MRC file, even after swapping bytes in header.".into(),
            );
        }
        true
    };
    let (nx, ny, nz, mode, next) = (
        values[0] as usize,
        values[1] as usize,
        values[2] as usize,
        values[3],
        values[7] as usize,
    );
    let (size, factor) = match mode {
        MRC_MODE_BYTE => (1, 1),
        MRC_MODE_SHORT => (2, 1),
        MRC_MODE_FLOAT => (4, 1),
        MRC_MODE_COMPLEX_SHORT => (2, 2),
        MRC_MODE_COMPLEX_FLOAT => (4, 2),
        MRC_MODE_RGB => (1, 3),
        _ => return Err(format!("mrcx: data type {mode} unsupported.")),
    };
    let header_size = MRC_HEADER_SIZE
        .checked_add(next)
        .ok_or_else(|| "MRC extended header size overflow".to_string())?;
    if input.len() < header_size {
        return Err("input file ends in its extended header".into());
    }
    let mut converted = convert_header(&input, next, direction, options.format)?;
    converted.extend_from_slice(&convert_body(
        &input[header_size..],
        nx,
        ny,
        nz,
        size,
        factor,
        direction,
        options.format,
    )?);
    let destination = options.output.as_deref().unwrap_or(&options.input);
    fs::write(Path::new(destination), converted)
        .map_err(|error| format!("mrcx: Couldn't open {destination}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_swaps_match_source() {
        let mut out = [0; 4];
        convert_longs(&[1, 2, 3, 4], &mut out, 1, false).unwrap();
        assert_eq!(out, [4, 3, 2, 1]);
        convert_shorts(&[1, 2, 3, 4], &mut out, 2, false).unwrap();
        assert_eq!(out, [2, 1, 4, 3]);
    }
    #[test]
    fn converts_a_tiny_big_endian_short_mrc() {
        let mut file = vec![0u8; 1024 + 4];
        for (offset, value) in [
            (0, 2i32),
            (4, 1),
            (8, 1),
            (12, 1),
            (28, 2),
            (32, 1),
            (36, 1),
            (64, 1),
            (68, 2),
            (72, 3),
            (92, 0),
        ] {
            file[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
        }
        file[208..212].copy_from_slice(b"MAP ");
        file[212] = 17;
        file[1024..].copy_from_slice(&[0, 1, 0, 2]);
        let converted = {
            let h = convert_header(&file, 0, true, MrcxFloatFormat::Ieee).unwrap();
            let mut o = h;
            o.extend_from_slice(
                &convert_body(&file[1024..], 2, 1, 1, 2, 1, true, MrcxFloatFormat::Ieee).unwrap(),
            );
            o
        };
        assert_eq!(i32::from_ne_bytes(converted[0..4].try_into().unwrap()), 2);
        assert_eq!(&converted[1024..], &[1, 0, 2, 0]);
        assert_eq!(converted[212], 68);
    }
}
