//! Owned translation of `IMOD/mrc/raw2mrc.c`.

use std::path::{Path, PathBuf};

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_RGB, MRC_MODE_SHORT, MRC_MODE_USHORT, MrcHeader,
    mrc_head_label, mrc_head_new, mrc_head_write, mrc_set_scale, mrc_write_slice,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RawDataType {
    Byte,
    SignedByte,
    Rgb,
    Short,
    UnsignedShort,
    Long,
    UnsignedLong,
    LongToShort,
    UnsignedLongToShort,
    Float,
    Double,
    DoubleToShort,
}
impl RawDataType {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "byte" => Self::Byte,
            "sbyte" => Self::SignedByte,
            "rgb" => Self::Rgb,
            "short" => Self::Short,
            "ushort" => Self::UnsignedShort,
            "long" => Self::Long,
            "ulong" => Self::UnsignedLong,
            "long2short" => Self::LongToShort,
            "ulong2short" => Self::UnsignedLongToShort,
            "float" => Self::Float,
            "double" => Self::Double,
            "double2short" => Self::DoubleToShort,
            _ => return None,
        })
    }
    fn bytes(self) -> usize {
        match self {
            Self::Byte | Self::SignedByte => 1,
            Self::Rgb => 3,
            Self::Short | Self::UnsignedShort => 2,
            Self::Long
            | Self::UnsignedLong
            | Self::LongToShort
            | Self::UnsignedLongToShort
            | Self::Float => 4,
            Self::Double | Self::DoubleToShort => 8,
        }
    }
    fn mode(self, keep_unsigned: bool) -> i32 {
        match self {
            Self::Rgb => MRC_MODE_RGB,
            Self::Short | Self::LongToShort | Self::DoubleToShort => MRC_MODE_SHORT,
            Self::UnsignedShort | Self::UnsignedLongToShort if keep_unsigned => MRC_MODE_USHORT,
            Self::UnsignedShort | Self::UnsignedLongToShort => MRC_MODE_SHORT,
            Self::Long | Self::UnsignedLong | Self::Float | Self::Double => MRC_MODE_FLOAT,
            _ => MRC_MODE_BYTE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Raw2MrcOptions {
    pub inputs: Vec<PathBuf>,
    pub output: PathBuf,
    pub width: usize,
    pub height: usize,
    pub sections_per_file: usize,
    pub data_type: RawDataType,
    pub swap_bytes: bool,
    pub header_bytes: usize,
    pub skip_between_sections: usize,
    pub flip_y: bool,
    pub invert_files: bool,
    pub invert_sections: bool,
    pub divide_by_two: bool,
    pub keep_unsigned: bool,
    pub convert_to_short: bool,
    pub pixel_spacing: Option<f64>,
    pub z_pixel_spacing: Option<f64>,
}
impl Raw2MrcOptions {
    pub fn default_for(inputs: Vec<PathBuf>, output: PathBuf) -> Self {
        Self {
            inputs,
            output,
            width: 640,
            height: 480,
            sections_per_file: 1,
            data_type: RawDataType::Byte,
            swap_bytes: false,
            header_bytes: 0,
            skip_between_sections: 0,
            flip_y: false,
            invert_files: false,
            invert_sections: false,
            divide_by_two: false,
            keep_unsigned: false,
            convert_to_short: false,
            pixel_spacing: None,
            z_pixel_spacing: None,
        }
    }
}

pub fn rawswap(data: &mut [u8], pixel_size: usize) {
    if matches!(pixel_size, 2 | 4) {
        for pixel in data.chunks_exact_mut(pixel_size) {
            pixel.reverse();
        }
    }
}
/// `rawdecompress` is a source no-op: only `CTYPE_NONE` exists in this command.
pub fn rawdecompress(_: &mut [u8], _: usize, _: usize, _: i32) {}

fn output_type(options: &Raw2MrcOptions) -> RawDataType {
    if !options.convert_to_short {
        return options.data_type;
    }
    match options.data_type {
        RawDataType::Long => RawDataType::LongToShort,
        RawDataType::UnsignedLong => RawDataType::UnsignedLongToShort,
        RawDataType::Double => RawDataType::DoubleToShort,
        other => other,
    }
}
fn number(bytes: &[u8], ty: RawDataType) -> f64 {
    match ty {
        RawDataType::Byte | RawDataType::Rgb => bytes[0] as f64,
        RawDataType::SignedByte => bytes[0] as i8 as f64,
        RawDataType::Short => i16::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        RawDataType::UnsignedShort => u16::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        RawDataType::Long | RawDataType::LongToShort => {
            i32::from_ne_bytes(bytes.try_into().unwrap()) as f64
        }
        RawDataType::UnsignedLong | RawDataType::UnsignedLongToShort => {
            u32::from_ne_bytes(bytes.try_into().unwrap()) as f64
        }
        RawDataType::Float => f32::from_ne_bytes(bytes.try_into().unwrap()) as f64,
        RawDataType::Double | RawDataType::DoubleToShort => {
            f64::from_ne_bytes(bytes.try_into().unwrap())
        }
    }
}
fn encode(value: f64, mode: i32, out: &mut Vec<u8>) {
    match mode {
        MRC_MODE_BYTE => out.push(value as u8),
        MRC_MODE_SHORT => out.extend_from_slice(&(value as i16).to_ne_bytes()),
        MRC_MODE_USHORT => out.extend_from_slice(&(value as u16).to_ne_bytes()),
        MRC_MODE_FLOAT => out.extend_from_slice(&(value as f32).to_ne_bytes()),
        _ => unreachable!(),
    }
}
fn convert_slice(
    input: &[u8],
    ty: RawDataType,
    mode: i32,
    divide: bool,
    keep_unsigned: bool,
) -> Vec<u8> {
    if ty == RawDataType::Rgb {
        return input.to_vec();
    }
    let mut out = Vec::with_capacity(input.len().max(input.len() * 4 / ty.bytes()));
    for source in input.chunks_exact(ty.bytes()) {
        let raw = number(source, ty);
        let value = match ty {
            RawDataType::SignedByte => raw + 128.0,
            RawDataType::UnsignedShort => {
                if divide {
                    raw / 2.0
                } else if keep_unsigned {
                    raw
                } else {
                    raw - 32767.0
                }
            }
            RawDataType::UnsignedLongToShort => {
                if divide {
                    raw / 2.0
                } else if keep_unsigned {
                    raw
                } else {
                    raw - 32767.0
                }
            }
            RawDataType::LongToShort | RawDataType::DoubleToShort if divide => raw / 2.0,
            _ => raw,
        };
        encode(value, mode, &mut out);
    }
    out
}
fn stats(slice: &[u8], mode: i32) -> (f32, f32, f64) {
    let bytes = match mode {
        MRC_MODE_BYTE => 1,
        MRC_MODE_SHORT | MRC_MODE_USHORT => 2,
        MRC_MODE_FLOAT => 4,
        _ => return (0.0, 0.0, 0.0),
    };
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0.0;
    let mut count = 0usize;
    for bytes in slice.chunks_exact(bytes) {
        let value = match mode {
            MRC_MODE_BYTE => bytes[0] as f64,
            MRC_MODE_SHORT => i16::from_ne_bytes(bytes.try_into().unwrap()) as f64,
            MRC_MODE_USHORT => u16::from_ne_bytes(bytes.try_into().unwrap()) as f64,
            MRC_MODE_FLOAT => f32::from_ne_bytes(bytes.try_into().unwrap()) as f64,
            _ => unreachable!(),
        };
        minimum = minimum.min(value as f32);
        maximum = maximum.max(value as f32);
        sum += value;
        count += 1;
    }
    (minimum, maximum, sum / count as f64)
}

/// Complete conversion body of C `main`, returning its printed statistics.
pub fn convert_raw_to_mrc(options: &Raw2MrcOptions) -> Result<(f32, f32, f32), String> {
    if options.inputs.is_empty()
        || options.width == 0
        || options.height == 0
        || options.sections_per_file == 0
    {
        return Err("input files, X/Y size, and sections per file are required".into());
    }
    let data_type = output_type(options);
    if options.divide_by_two && options.keep_unsigned && data_type != RawDataType::DoubleToShort {
        return Err("cannot divide by 2 and keep mode as unsigned".into());
    }
    let mode = data_type.mode(options.keep_unsigned);
    let pixels = options
        .width
        .checked_mul(options.height)
        .ok_or("image size overflow")?;
    let sections = options
        .sections_per_file
        .checked_mul(options.inputs.len())
        .ok_or("section count overflow")?;
    let mut header = MrcHeader::default();
    mrc_head_new(
        &mut header,
        options.width as i32,
        options.height as i32,
        sections as i32,
        mode,
    );
    if let Some(pixel) = options.pixel_spacing.filter(|value| *value > 0.0) {
        mrc_set_scale(
            &mut header,
            pixel,
            pixel,
            options.z_pixel_spacing.unwrap_or(pixel),
        );
    }
    let mut output = ImodFile::open(&options.output, "wb").ok_or("opening output file")?;
    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    let mut mean = 0.0f64;
    let mut written = 0;
    let file_order: Vec<_> = if options.invert_files {
        options.inputs.iter().rev().collect()
    } else {
        options.inputs.iter().collect()
    };
    for input_path in file_order {
        let file = std::fs::read(input_path)
            .map_err(|_| format!("opening input {}", input_path.display()))?;
        for section in 0..options.sections_per_file {
            let read_section = if options.invert_sections {
                options.sections_per_file - 1 - section
            } else {
                section
            };
            let start = options.header_bytes
                + read_section * (options.skip_between_sections + pixels * data_type.bytes());
            let end = start
                .checked_add(pixels * data_type.bytes())
                .ok_or("input offset overflow")?;
            let mut raw = file
                .get(start..end)
                .ok_or_else(|| format!("reading data from {}", input_path.display()))?
                .to_vec();
            if options.swap_bytes {
                rawswap(&mut raw, data_type.bytes());
            }
            let mut slice = convert_slice(
                &raw,
                data_type,
                mode,
                options.divide_by_two,
                options.keep_unsigned,
            );
            let (min, max, section_mean) = stats(&slice, mode);
            if options.flip_y {
                let row = slice.len() / options.height;
                for y in 0..options.height / 2 {
                    let lower = y * row;
                    let upper = (options.height - 1 - y) * row;
                    for offset in 0..row {
                        slice.swap(lower + offset, upper + offset);
                    }
                }
            }
            header.amin = min;
            header.amax = max;
            if mrc_write_slice(&slice, &mut output, &mut header, written, b'Z') != 0 {
                return Err("writing data to file".into());
            }
            written += 1;
            mean += section_mean;
            global_min = global_min.min(min);
            global_max = global_max.max(max);
        }
    }
    header.amin = global_min;
    header.amax = global_max;
    header.amean = (mean / written as f64) as f32;
    mrc_head_label(&mut header, b"raw2mrc: Converted to mrc format.");
    if mrc_head_write(&mut output, &mut header) != 0 {
        return Err("writing MRC header".into());
    }
    Ok((global_min, global_max, header.amean))
}

/// Safe command entry point corresponding to C `main`.  It accepts the
/// source's long option names and short aliases, with files supplied by
/// `-input` or positionally and the output supplied by `-output` or last.
pub fn raw2mrc(arguments: &[String]) -> Result<(f32, f32, f32), String> {
    let mut inputs = Vec::new();
    let mut positional = Vec::new();
    let mut output = None;
    let mut options = Raw2MrcOptions::default_for(Vec::new(), PathBuf::new());
    let mut index = 1usize;
    while index < arguments.len() {
        let argument = &arguments[index];
        let value = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            arguments
                .get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("missing value for {argument}"))
        };
        match argument.as_str() {
            "-input" | "--input" => inputs.push(PathBuf::from(value(&mut index)?)),
            "-output" | "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "-x" | "--x" => {
                options.width = value(&mut index)?.parse().map_err(|_| "invalid X size")?
            }
            "-y" | "--y" => {
                options.height = value(&mut index)?.parse().map_err(|_| "invalid Y size")?
            }
            "-z" | "--z" => {
                options.sections_per_file = value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid section count")?
            }
            "-t" | "--type" => {
                options.data_type =
                    RawDataType::parse(value(&mut index)?).ok_or("invalid data type")?
            }
            "-s" | "--swap" => options.swap_bytes = true,
            "-o" | "--offset" => {
                options.header_bytes = value(&mut index)?.parse().map_err(|_| "invalid offset")?
            }
            "-oz" | "--skip" => {
                options.skip_between_sections = value(&mut index)?
                    .parse()
                    .map_err(|_| "invalid section skip")?
            }
            "-f" | "--flip" => options.flip_y = true,
            "-i" | "--invert" => options.invert_files = true,
            "-iz" | "--invert-sections" => options.invert_sections = true,
            "-d" | "--divide" => options.divide_by_two = true,
            "-u" | "--unsigned" => options.keep_unsigned = true,
            "-c" | "--convert" => options.convert_to_short = true,
            "-p" | "--pixel-spacing" => {
                options.pixel_spacing = Some(
                    value(&mut index)?
                        .parse()
                        .map_err(|_| "invalid pixel spacing")?,
                )
            }
            "-pz" | "--z-pixel-spacing" => {
                options.z_pixel_spacing = Some(
                    value(&mut index)?
                        .parse()
                        .map_err(|_| "invalid Z pixel spacing")?,
                )
            }
            "-help" | "--help" => {
                return Err("usage: raw2mrc [options] input-files output-file".into());
            }
            value if value.starts_with('-') => return Err(format!("unknown option {value}")),
            value => positional.push(PathBuf::from(value)),
        }
        index += 1;
    }
    if output.is_none() {
        output = positional.pop();
    }
    inputs.extend(positional);
    options.inputs = inputs;
    options.output = output.ok_or("an output filename must be entered")?;
    convert_raw_to_mrc(&options)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swaps_exact_pixels() {
        let mut data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        rawswap(&mut data, 4);
        assert_eq!(data, vec![4, 3, 2, 1, 8, 7, 6, 5]);
    }
    #[test]
    fn converts_real_raw_fixture_to_mrc() {
        let directory = std::env::temp_dir();
        let raw = directory.join("raw2mrc-rust-test.raw");
        let mrc = directory.join("raw2mrc-rust-test.mrc");
        std::fs::write(&raw, [1u8, 2, 3, 4]).unwrap();
        let mut options = Raw2MrcOptions::default_for(vec![raw.clone()], mrc.clone());
        options.width = 2;
        options.height = 2;
        let stats = convert_raw_to_mrc(&options).unwrap();
        assert_eq!(stats, (1.0, 4.0, 2.5));
        assert_eq!(std::fs::metadata(&mrc).unwrap().len(), 1028);
        let _ = std::fs::remove_file(raw);
        let _ = std::fs::remove_file(mrc);
    }
}
