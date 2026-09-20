//! Safe numerical translation of `IMOD/mrc/ctfphaseflip.cpp`.
//!
//! The source's very large `main` combines Pip parsing, MRC/HDF scheduling,
//! GPU selection, and its actual CTF phase-only Fourier correction.  This
//! module keeps those data choices owned and exposes the correction kernel for
//! the crate MRC layer to drive.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libcfshr::b3dutil::angle_within_limits;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MrcHeader, mrc_head_label, mrc_head_read,
    mrc_head_write, mrc_init_output_header, mrc_read_slice, mrc_write_slice,
};
use rustfft::{FftPlanner, num_complex::Complex32};

pub const UNUSED_DEFOCUS: f32 = -2_000_000.;
pub const MIN_ANGLE: f32 = 0.001;
const MY_PI: f64 = 3.1415926;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SavedDefocus {
    pub starting_slice: i32,
    pub ending_slice: i32,
    pub low_angle: f64,
    pub high_angle: f64,
    pub defocus: f64,
    pub defocus2: f64,
    pub astig_angle: f64,
    pub plate_phase: f64,
    pub cut_on_frequency: f64,
}
pub const DEF_FILE_HAS_ASTIG: i32 = 1;
pub const DEF_FILE_ASTIG_IN_RAD: i32 = 2;
pub const DEF_FILE_HAS_PHASE: i32 = 4;
pub const DEF_FILE_PHASE_IN_RAD: i32 = 8;
pub const DEF_FILE_INVERT_ANGLES: i32 = 16;
pub const DEF_FILE_HAS_CUT_ON: i32 = 32;

/// C `getAnyOneZero`: CTF zero in Nyquist units.
pub fn get_any_one_zero(
    defocus: f64,
    phase: f64,
    zero_num: i32,
    amp_contrast: f64,
    cs: f64,
    pixel_size: f64,
    voltage: f64,
) -> f64 {
    let wavelength = 1.23984 / (voltage * (voltage + 1022.)).sqrt();
    let cs_one = (cs * wavelength).sqrt();
    let cs_two = (1_000_000. * cs / wavelength).sqrt().sqrt();
    let amp_angle = 2. * (amp_contrast / (1. - amp_contrast * amp_contrast).sqrt()).atan() / MY_PI;
    let delta = defocus / cs_one;
    let theta = (delta
        - (0f64)
            .max(delta * delta + amp_angle + 2. * phase / MY_PI - 2. * zero_num as f64)
            .sqrt())
    .sqrt();
    theta * pixel_size * 2. / (wavelength * cs_two)
}

/// C `firstZeroShift`.
pub fn first_zero_shift(
    defocus: f64,
    phase: f64,
    angle: f64,
    extent: i32,
    amp_contrast: f64,
    cs: f64,
    pixel_size: f64,
    voltage: f64,
) -> f64 {
    let low = defocus - 0.5 * extent as f64 * angle.tan() * pixel_size / 1000.;
    (get_any_one_zero(low, phase, 1, amp_contrast, cs, pixel_size, voltage)
        - get_any_one_zero(
            defocus + 0.5 * extent as f64 * angle.tan() * pixel_size / 1000.,
            phase,
            1,
            amp_contrast,
            cs,
            pixel_size,
            voltage,
        ))
    .abs()
}

/// C `interpolateTable`, including the special angular shortest-path rule.
pub fn interpolate_table(defocus: &mut [f32], if_angles: bool) {
    let mut first = None;
    let mut second = 0;
    for k in 0..defocus.len() {
        if defocus[k] == UNUSED_DEFOCUS {
            continue;
        }
        second = k;
        if let Some(previous) = first {
            for row in previous + 1..second {
                if if_angles {
                    let difference =
                        angle_within_limits(defocus[second] - defocus[previous], -90., 90.) as f32;
                    defocus[row] = angle_within_limits(
                        defocus[previous]
                            + (row - previous) as f32 * difference / (second - previous) as f32,
                        -90.,
                        90.,
                    ) as f32;
                } else {
                    defocus[row] = ((row - previous) as f32 * defocus[second]
                        + (second - row) as f32 * defocus[previous])
                        / (second - previous) as f32;
                }
            }
        } else {
            for row in 0..second {
                defocus[row] = defocus[second];
            }
        }
        first = Some(second);
    }
    if first.is_some() {
        for row in (0..defocus.len()).rev() {
            if defocus[row] == UNUSED_DEFOCUS {
                defocus[row] = defocus[second];
            } else {
                break;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtfPhaseFlipParameters {
    pub pixel_size_nm: f32,
    pub voltage_kv: f32,
    pub spherical_aberration_mm: f32,
    pub amplitude_contrast: f32,
    pub phase_shift: f32,
    pub cut_on_frequency: f32,
    pub scale_by_ctf_power: f32,
}
impl CtfPhaseFlipParameters {
    pub fn ctf(
        &self,
        fx: f32,
        fy: f32,
        defocus_microns: f32,
        astig_defocus_microns: f32,
        astig_angle_radians: f32,
    ) -> f32 {
        let wavelength = 12.3984 / (self.voltage_kv * (self.voltage_kv + 1022.)).sqrt();
        let c1 = core::f32::consts::PI * wavelength;
        let c2 = c1 * self.spherical_aberration_mm * 1.0e7 * wavelength * wavelength / 2.;
        let frequency_sq = fx * fx + fy * fy;
        let angle = fy.atan2(fx) - astig_angle_radians;
        let defocus = if astig_defocus_microns > 0. {
            defocus_microns + (astig_defocus_microns - defocus_microns) * angle.sin().powi(2)
        } else {
            defocus_microns
        };
        let phase = c1 * defocus * 10_000. * frequency_sq - c2 * frequency_sq * frequency_sq
            + self.phase_shift;
        let mut ctf = -(phase.sin()
            + self.amplitude_contrast
                / (1. - self.amplitude_contrast * self.amplitude_contrast).sqrt()
                * phase.cos());
        if self.cut_on_frequency > 0. {
            ctf *= 1. - (-frequency_sq.sqrt() / self.cut_on_frequency).exp();
        }
        ctf
    }
}

/// CPU counterpart of the source FFT/correction/FFT block for one full image.
/// It phase-flips all Fourier samples and supports the source CTF-power scale.
pub fn ctf_phase_flip(
    image: &[f32],
    nx: usize,
    ny: usize,
    defocus: f32,
    defocus2: f32,
    astig_angle: f32,
    parameters: CtfPhaseFlipParameters,
) -> Result<Vec<f32>, String> {
    if nx == 0 || ny == 0 || image.len() != nx * ny {
        return Err("image dimensions do not match pixel data".into());
    }
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(nx);
    let inverse = planner.plan_fft_inverse(nx);
    let mut rows: Vec<Vec<Complex32>> = (0..ny)
        .map(|y| {
            image[y * nx..(y + 1) * nx]
                .iter()
                .map(|&v| Complex32::new(v, 0.))
                .collect()
        })
        .collect();
    for row in &mut rows {
        forward.process(row);
    }
    let column_forward = planner.plan_fft_forward(ny);
    let column_inverse = planner.plan_fft_inverse(ny);
    for x in 0..nx {
        let mut column: Vec<Complex32> = (0..ny).map(|y| rows[y][x]).collect();
        column_forward.process(&mut column);
        for y in 0..ny {
            rows[y][x] = column[y];
        }
    }
    for y in 0..ny {
        let fy = if y <= ny / 2 {
            y as f32 / ny as f32
        } else {
            (y as f32 - ny as f32) / ny as f32
        };
        for x in 0..nx {
            let fx = if x <= nx / 2 {
                x as f32 / nx as f32
            } else {
                (x as f32 - nx as f32) / nx as f32
            };
            let ctf = parameters.ctf(fx, fy, defocus, defocus2, astig_angle);
            let sign = if ctf < 0. { -1. } else { 1. };
            let power = parameters.scale_by_ctf_power;
            if power > 0. {
                rows[y][x] *= sign * ctf.abs().powf(power)
            } else {
                rows[y][x] *= sign;
            }
        }
    }
    for x in 0..nx {
        let mut column: Vec<Complex32> = (0..ny).map(|y| rows[y][x]).collect();
        column_inverse.process(&mut column);
        for y in 0..ny {
            rows[y][x] = column[y];
        }
    }
    for row in &mut rows {
        inverse.process(row);
    }
    let scale = 1. / (nx * ny) as f32;
    Ok(rows.into_iter().flatten().map(|x| x.re * scale).collect())
}

#[derive(Clone, Debug, PartialEq)]
pub struct CtfPhaseFlipOptions {
    pub input: String,
    pub output: String,
    pub angle_file: Option<String>,
    pub defocus_file: String,
    pub interpolation_width: i32,
    pub pixel_size: f32,
    pub voltage: i32,
    pub spherical_aberration: f32,
    pub amplitude_contrast: f32,
    pub phase_shift: f32,
    pub scale_by_ctf_power: f32,
    pub invert_angles: bool,
}

/// Source `main` option validation in an owned command representation.
pub fn ctfphaseflip_options(arguments: &[String]) -> Result<CtfPhaseFlipOptions, String> {
    let mut values = std::collections::BTreeMap::<String, String>::new();
    let mut invert = false;
    let mut i = 1;
    while i < arguments.len() {
        let arg = &arguments[i];
        if !arg.starts_with('-') {
            i += 1;
            continue;
        }
        if arg == "-invert" {
            invert = true;
            i += 1;
            continue;
        }
        let key = arg.trim_start_matches('-').to_ascii_lowercase();
        i += 1;
        values.insert(
            key,
            arguments
                .get(i)
                .ok_or_else(|| format!("Missing value for {arg}"))?
                .clone(),
        );
        i += 1;
    }
    let take = |names: &[&str]| names.iter().find_map(|n| values.get(*n)).cloned();
    let input = take(&["input"]).ok_or("No stack specified")?;
    let output = take(&["output"]).ok_or("OutputFileName is not specified")?;
    let defocus_file =
        take(&["deffn", "defocusfile", "defocus"]).ok_or("No defocus file is specified")?;
    let parse = |names: &[&str], message: &str| -> Result<f32, String> {
        take(names)
            .ok_or_else(|| message.to_owned())?
            .parse()
            .map_err(|_| message.into())
    };
    Ok(CtfPhaseFlipOptions {
        input,
        output,
        angle_file: take(&["anglefn", "anglefile"]),
        defocus_file,
        interpolation_width: parse(
            &["iwidth", "interpolationwidth"],
            "No InterpolationWidth specified",
        )? as i32,
        pixel_size: parse(&["pixelsize"], "No PixelSize specified")?,
        voltage: parse(&["volt", "voltage"], "Voltage is not specified")? as i32,
        spherical_aberration: parse(
            &["cs", "sphericalaberration"],
            "SphericalAberration is not specified",
        )?,
        amplitude_contrast: parse(
            &["ampcontrast", "amplitudecontrast"],
            "No AmplitudeContrast is specified",
        )?,
        phase_shift: take(&["phase", "phaseplateshift"])
            .unwrap_or_else(|| "0".into())
            .parse()
            .map_err(|_| "bad phase shift")?,
        scale_by_ctf_power: take(&["scale", "scalebyctfpower"])
            .unwrap_or_else(|| "0".into())
            .parse()
            .map_err(|_| "bad CTF power")?,
        invert_angles: invert,
    })
}

/// Owned executable equivalent of the ordinary (non-GPU, non-parallel-HDF)
/// path through C `main`.  The caller supplies the parsed defocus records;
/// parsing ctfplotter's text format remains with the translated `ctfutils`
/// boundary rather than making this command depend on C `Ilist` ownership.
pub fn ctfphaseflip(options: &CtfPhaseFlipOptions, records: &[SavedDefocus]) -> Result<(), String> {
    let mut input = ImodFile::open(&options.input, "rb")
        .ok_or_else(|| format!("Could not open input file {}", options.input))?;
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut header) != 0 {
        return Err("Reading header of input stack".into());
    }
    if !matches!(header.mode, MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_FLOAT) {
        return Err(format!(
            "File mode is {}; only byte, short integer, or real allowed",
            header.mode
        ));
    }
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;
    let mut defocus = vec![UNUSED_DEFOCUS; nz];
    let mut defocus2 = vec![UNUSED_DEFOCUS; nz];
    let mut angles = vec![0.; nz];
    let mut phase = vec![0.; nz];
    for record in records {
        let slice = ((record.starting_slice + record.ending_slice) / 2) as usize;
        if slice >= nz {
            return Err("View numbers in defocus file are out of range".into());
        }
        defocus[slice] = record.defocus as f32;
        defocus2[slice] = record.defocus2 as f32;
        angles[slice] = record.astig_angle as f32;
        phase[slice] = record.plate_phase as f32;
    }
    interpolate_table(&mut defocus, false);
    interpolate_table(&mut defocus2, false);
    interpolate_table(&mut angles, true);
    interpolate_table(&mut phase, false);
    if defocus.iter().any(|value| *value == UNUSED_DEFOCUS) {
        return Err("specified defocus is wrong for one or more slices".into());
    }
    let mut output = ImodFile::open(&options.output, "wb")
        .ok_or_else(|| format!("Could not open output file {}", options.output))?;
    let mut out_header = header.clone();
    out_header.mode = MRC_MODE_FLOAT;
    mrc_init_output_header(&mut out_header);
    mrc_head_label(
        &mut out_header,
        b"ctfPhaseFlip: CTF correction with phase flipping only",
    );
    let parameters = CtfPhaseFlipParameters {
        pixel_size_nm: options.pixel_size,
        voltage_kv: options.voltage as f32,
        spherical_aberration_mm: options.spherical_aberration.max(0.01),
        amplitude_contrast: options.amplitude_contrast,
        phase_shift: options.phase_shift,
        cut_on_frequency: 0.,
        scale_by_ctf_power: options.scale_by_ctf_power,
    };
    let bytes_per_pixel = match header.mode {
        MRC_MODE_BYTE => 1,
        MRC_MODE_SHORT => 2,
        MRC_MODE_FLOAT => 4,
        _ => unreachable!(),
    };
    for z in 0..nz {
        let mut raw = vec![0; nx * ny * bytes_per_pixel];
        if mrc_read_slice(&mut raw, &mut input, &mut header, z as i32, b'Z') != 0 {
            return Err(format!("Reading slice {}", z + 1));
        };
        let image: Vec<f32> = match header.mode {
            MRC_MODE_BYTE => raw.iter().map(|&x| x as f32).collect(),
            MRC_MODE_SHORT => raw
                .chunks_exact(2)
                .map(|x| i16::from_ne_bytes([x[0], x[1]]) as f32)
                .collect(),
            _ => raw
                .chunks_exact(4)
                .map(|x| f32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
                .collect(),
        };
        let mut p = parameters;
        p.phase_shift += phase[z];
        let corrected = ctf_phase_flip(
            &image,
            nx,
            ny,
            defocus[z],
            defocus2[z],
            angles[z].to_radians(),
            p,
        )?;
        let mut bytes = Vec::with_capacity(corrected.len() * 4);
        for value in corrected {
            bytes.extend_from_slice(&value.to_ne_bytes())
        }
        if mrc_write_slice(&bytes, &mut output, &mut out_header, z as i32, b'Z') != 0 {
            return Err(format!("Writing slice {}", z + 1));
        };
    }
    if mrc_head_write(&mut output, &mut out_header) != 0 {
        return Err("Writing slice header error".into());
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_and_interpolation_are_source_shaped() {
        assert!(get_any_one_zero(3., 0., 1, 0.07, 2.7, 0.5, 300.).is_finite());
        let mut a = [UNUSED_DEFOCUS, 1., UNUSED_DEFOCUS, 3., UNUSED_DEFOCUS];
        interpolate_table(&mut a, false);
        assert_eq!(a, [1., 1., 2., 3., 3.]);
    }
    #[test]
    fn constant_image_phase_flip_is_finite() {
        let p = CtfPhaseFlipParameters {
            pixel_size_nm: 1.,
            voltage_kv: 300.,
            spherical_aberration_mm: 2.7,
            amplitude_contrast: 0.07,
            phase_shift: 0.,
            cut_on_frequency: 0.,
            scale_by_ctf_power: 0.,
        };
        assert!(
            ctf_phase_flip(&[1.; 16], 4, 4, 3., 0., 0., p)
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
    }
}
