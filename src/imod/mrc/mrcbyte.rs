//! Translation of `IMOD/mrc/mrcbyte.c`.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{
    LoadInfo, MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MRC_MODE_USHORT, MRC_RAMP_EXP,
    MRC_RAMP_LIN, MRC_RAMP_LOG, MrcHeader, mrc_contrast_scaling, mrc_get_scale, mrc_head_label,
    mrc_head_read, mrc_head_write, mrc_init_li, mrc_init_output_header, mrc_set_scale,
    mrc_write_slice,
};
use crate::imod::libiimod::mrcsec::mrc_read_z_byte;

pub fn mrcbyte_help(name: &str) {
    println!("Usage: {name} -[rRblecsxyz] <infile> <outfile>");
    println!(
        "Options:\n\t-r\tRequest user input for size and contrast info.\n\t-c blk,wht\tContrast scaling with black and white levels\n\t-s min,max\tInitial intensity scaling (min to 0, max to 255)\n\t-R\tReverse contrast in output.\n\t-l\tUse logarithmic ramp.\n\t-e\tUse exponential ramp.\n\t-x min,max\tWrite subset of image in X\n\t-y min,max\tWrite subset of image in Y\n\t-z min,max\tWrite subset of image in Z\n\t-b\tWrite raw byte data, no header"
    );
}

/// C `main` in `mrcbyte.c`.
pub fn mrcbyte(arguments: &[String]) -> i32 {
    let program = arguments.first().map_or("mrcbyte", String::as_str);
    if arguments.len() < 3 {
        mrcbyte_help(program);
        return 3;
    }
    let mut li = LoadInfo::default();
    mrc_init_li(Some(&mut li), None);
    let mut data_only = false;
    let mut reverse = false;
    let mut ramp = MRC_RAMP_LIN;
    let mut resize = false;
    let mut index = 1;
    while index < arguments.len() && arguments[index].starts_with('-') && arguments[index] != "-" {
        let option = &arguments[index];
        let key = option.as_bytes().get(1).copied();
        let value = |index: &mut usize| -> Option<&str> {
            if option.len() > 2 {
                Some(&option[2..])
            } else {
                *index += 1;
                arguments.get(*index).map(String::as_str)
            }
        };
        match key {
            Some(b'b') if option.len() == 2 => data_only = true,
            Some(b'r') if option.len() == 2 => resize = true,
            Some(b'R') if option.len() == 2 => reverse = true,
            Some(b'l') if option.len() == 2 => ramp = MRC_RAMP_LOG,
            Some(b'e') if option.len() == 2 => ramp = MRC_RAMP_EXP,
            Some(b'c') => {
                let Some(text) = value(&mut index) else {
                    return 1;
                };
                let Some((black, white)) = text
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                else {
                    eprintln!("ERROR: {program} - invalid contrast range");
                    return 1;
                };
                li.black = black;
                li.white = white;
            }
            Some(b's') => {
                let Some(text) = value(&mut index) else {
                    return 1;
                };
                let Some((minimum, maximum)) = text
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                else {
                    eprintln!("ERROR: {program} - invalid scaling range");
                    return 1;
                };
                li.smin = minimum;
                li.smax = maximum;
            }
            Some(b'x') | Some(b'y') | Some(b'z') => {
                let Some(text) = value(&mut index) else {
                    return 1;
                };
                let Some((minimum, maximum)) = text
                    .split_once(',')
                    .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
                else {
                    eprintln!("ERROR: {program} - invalid subset range");
                    return 1;
                };
                match key {
                    Some(b'x') => {
                        li.xmin = minimum;
                        li.xmax = maximum;
                    }
                    Some(b'y') => {
                        li.ymin = minimum;
                        li.ymax = maximum;
                    }
                    _ => {
                        li.zmin = minimum;
                        li.zmax = maximum;
                    }
                }
            }
            _ => {
                eprintln!("ERROR: {program} - Illegal option {option}");
                mrcbyte_help(program);
                return 1;
            }
        }
        index += 1;
    }
    if arguments.len() - index != 2 {
        mrcbyte_help(program);
        return 3;
    }
    if resize {
        eprintln!(
            "ERROR: {program} - interactive resize is unsupported in the non-interactive Rust command"
        );
        return 1;
    }
    let Some(mut input) = ImodFile::open(&arguments[index], "rb") else {
        eprintln!("ERROR: {program} - Opening {}.", arguments[index]);
        return 1;
    };
    let Some(mut output) = ImodFile::open(&arguments[index + 1], "wb") else {
        eprintln!("ERROR: {program} - Opening {}.", arguments[index + 1]);
        return 1;
    };
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut header) != 0 {
        eprintln!("ERROR: {program} - Reading input file header.");
        return 1;
    }
    if ramp != MRC_RAMP_LIN
        && !matches!(
            header.mode,
            MRC_MODE_SHORT | MRC_MODE_USHORT | MRC_MODE_FLOAT
        )
    {
        eprintln!(
            "ERROR: {program} - Nonlinear scaling can be used only with integer and float data"
        );
        return 1;
    }
    li.ramp = ramp;
    mrc_init_li(Some(&mut li), Some(&header));
    let (slope, offset) =
        mrc_contrast_scaling(&header, li.smin, li.smax, li.black, li.white, li.ramp);
    li.slope = slope;
    li.offset = offset;
    let mut output_header = header.clone();
    output_header.amin = 0.;
    output_header.amax = 255.;
    output_header.mode = MRC_MODE_BYTE;
    mrc_init_output_header(&mut output_header);
    output_header.nx = li.xmax + 1 - li.xmin;
    output_header.ny = li.ymax + 1 - li.ymin;
    output_header.nz = li.zmax + 1 - li.zmin;
    let (xscale, yscale, zscale) = mrc_get_scale(&header);
    mrc_set_scale(
        &mut output_header,
        xscale as f64,
        yscale as f64,
        zscale as f64,
    );
    mrc_head_label(
        &mut output_header,
        b"mrcbyte: Converted and scaled to byte mode.",
    );
    let area = output_header.nx as usize * output_header.ny as usize;
    let mut sum = 0.;
    if !data_only && mrc_head_write(&mut output, &mut output_header) != 0 {
        return 1;
    }
    for section in 0..output_header.nz {
        let mut bytes = vec![0; area];
        let input_section = section + li.zmin;
        if mrc_read_z_byte(&mut header, &mut li, &mut bytes, input_section) != 0 {
            eprintln!("ERROR: {program} - Reading section {input_section}");
            return 1;
        }
        if reverse {
            for byte in &mut bytes {
                *byte = 255 - *byte;
            }
        }
        let minimum = *bytes.iter().min().unwrap_or(&0) as f32;
        let maximum = *bytes.iter().max().unwrap_or(&0) as f32;
        let mean = bytes.iter().map(|value| *value as f64).sum::<f64>() / area as f64;
        if section == 0 {
            output_header.amin = minimum;
            output_header.amax = maximum;
        } else {
            output_header.amin = output_header.amin.min(minimum);
            output_header.amax = output_header.amax.max(maximum);
        }
        sum += mean;
        if mrc_write_slice(&bytes, &mut output, &mut output_header, section, b'Z') != 0 {
            return 1;
        }
    }
    output_header.amean = (sum / output_header.nz as f64) as f32;
    if !data_only && mrc_head_write(&mut output, &mut output_header) != 0 {
        return 1;
    };
    println!("Done");
    0
}

#[cfg(test)]
mod tests {
    use super::mrcbyte;
    use crate::imod::libcfshr::b3dutil::ImodFile;
    use crate::imod::libiimod::mrcfiles::{
        MRC_MODE_FLOAT, MrcHeader, mrc_head_new, mrc_head_read, mrc_head_write, mrc_write_slice,
    };
    #[test]
    fn converts_a_real_float_mrc_to_byte() {
        let root = std::env::temp_dir().join(format!("mrcbyte-{}", std::process::id()));
        let input = root.with_extension("in.mrc");
        let output = root.with_extension("out.mrc");
        let mut file = ImodFile::open(&input, "wb").unwrap();
        let mut header = MrcHeader::default();
        mrc_head_new(&mut header, 2, 1, 1, MRC_MODE_FLOAT);
        header.amin = 0.;
        header.amax = 10.;
        mrc_head_write(&mut file, &mut header);
        let data = [0_f32.to_ne_bytes(), 10_f32.to_ne_bytes()].concat();
        mrc_write_slice(&data, &mut file, &mut header, 0, b'Z');
        drop(file);
        assert_eq!(
            mrcbyte(&[
                "mrcbyte".into(),
                input.display().to_string(),
                output.display().to_string()
            ]),
            0
        );
        let mut file = ImodFile::open(&output, "rb").unwrap();
        let mut result = MrcHeader::default();
        assert_eq!(mrc_head_read(&mut file, &mut result), 0);
        assert_eq!(result.mode, 0);
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
