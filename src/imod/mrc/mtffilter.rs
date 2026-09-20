//! Owned numerical core of `mrc/mtffilter.cpp`.
//!
//! The command program is a composition of MRC I/O, padding, FFT execution and
//! these radial-frequency kernels.  Complex FFT samples are stored as adjacent
//! real/imaginary `f32` values, as in the source FFT buffers.

use crate::imod::raptor::main_classes::io_mrc_vol::{IoMrc, MrcVolume};
use std::collections::BTreeMap;

/// The source program's filter-selection state after command-line parsing.
#[derive(Clone, Debug, PartialEq)]
pub struct MtfFilterOptions {
    pub ctf: Vec<f32>,
    pub delta: f32,
    pub one_dimensional: bool,
    pub density_scale: f32,
}

/// Parsed source `PipReadOrParseOptions` values.  Repeated options retain all
/// words in source order; selection/metadata policy is handled by the caller.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MtfFilterCommand {
    pub values: BTreeMap<String, Vec<String>>,
}

impl MtfFilterCommand {
    /// Owned parsing of all 42 names/short aliases declared in mtffilter.cpp.
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let aliases = [
            ("input", "InputFile"),
            ("output", "OutputFile"),
            ("zrange", "StartingAndEndingZ"),
            ("mode", "ModeToOutput"),
            ("3dfilter", "FilterIn3D"),
            ("1dfilter", "OneDimensionalFilter"),
            ("units", "UnitsForFrequency"),
            ("lowpass", "LowPassRadiusSigma"),
            ("highpass", "HighPassSigma"),
            ("radius1", "FilterRadius1"),
            ("mtf", "MtfFile"),
            ("stock", "StockCurve"),
            ("maxinv", "MaximumInverse"),
            ("invrolloff", "InverseRolloffRadiusSigma"),
            ("xscale", "XScaleFactor"),
            ("noise", "NoisePadding"),
            ("denscale", "DensityScaleFactor"),
            ("rweight", "RWeightedFilter"),
            ("fake", "FakeSIRTiterations"),
            ("pixel", "PixelSize"),
            ("expanded", "ExpandedByFactor"),
            ("volt", "Voltage"),
            ("dtype", "TypeOfDoseFile"),
            ("dfile", "DoseWeightingFile"),
            ("dfixed", "FixedImageDose"),
            ("initial", "InitialDose"),
            ("bidir", "BidirectionalNumViews"),
            ("reversed", "ReversedBidirectional"),
            ("optimal", "OptimalDoseScaling"),
            ("critical", "CriticalDoseFactors"),
            ("verbose", "VerboseOutput"),
            ("deconv", "DeconvolutionStrength"),
            ("snr", "SNRFalloff"),
            ("dchigh", "HighPassNyquist"),
            ("defocus", "DefocusInMicrons"),
            ("dcphase", "PhaseShift"),
            ("cs", "SphericalAberration"),
            ("amplifier", "AmplifierFactorAndPower"),
            ("cutoff", "CutoffForAmplifier"),
            ("phase", "PhasePlateParameters"),
            ("param", "ParameterFile"),
            ("help", "usage"),
        ];
        let mut command = Self::default();
        let mut index = 0;
        while index < arguments.len() {
            let argument = &arguments[index];
            if !argument.starts_with('-') {
                return Err(format!("unexpected positional argument {argument}"));
            }
            let key = argument.trim_start_matches('-');
            let name = aliases
                .iter()
                .find(|(short, long)| key == *short || key.eq_ignore_ascii_case(long))
                .map(|(_, long)| *long)
                .ok_or_else(|| format!("unrecognized option {argument}"))?;
            command.values.entry(name.into()).or_default();
            index += 1;
            let arity = match name {
                "FilterIn3D"
                | "OneDimensionalFilter"
                | "NoisePadding"
                | "RWeightedFilter"
                | "ReversedBidirectional"
                | "usage" => 0,
                "StartingAndEndingZ"
                | "LowPassRadiusSigma"
                | "InverseRolloffRadiusSigma"
                | "AmplifierFactorAndPower" => 2,
                "CriticalDoseFactors" | "PhasePlateParameters" => 3,
                _ => 1,
            };
            if index + arity > arguments.len() {
                return Err(format!("not enough arguments for -{key}"));
            }
            command
                .values
                .get_mut(name)
                .expect("inserted option")
                .extend_from_slice(&arguments[index..index + arity]);
            index += arity;
        }
        command.validate()?;
        Ok(command)
    }
    pub fn has(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }
    pub fn values(&self, name: &str) -> Option<&[String]> {
        self.values.get(name).map(Vec::as_slice)
    }
    pub fn validate(&self) -> Result<(), String> {
        if !self.has("InputFile") && !self.has("usage") {
            return Err("No input file specified".into());
        }
        if self.has("FilterIn3D")
            && (self.has("OneDimensionalFilter")
                || self.has("RWeightedFilter")
                || self.has("FakeSIRTiterations"))
        {
            return Err(
                "3-D filtering cannot be combined with 1-D, R-weighted, or fake SIRT filtering"
                    .into(),
            );
        }
        if self.has("MtfFile") && self.has("StockCurve") {
            return Err("cannot enter both an MTF file and a stock curve".into());
        }
        if self.has("DeconvolutionStrength") && self.has("AmplifierFactorAndPower") {
            return Err("cannot combine deconvolution and amplifier filtering".into());
        }
        if self.has("FixedImageDose") && self.has("TypeOfDoseFile") {
            return Err("cannot enter both an image dose and type of dose file".into());
        }
        if (self.has("FixedImageDose") || self.has("TypeOfDoseFile"))
            && (self.has("FilterIn3D")
                || self.has("OneDimensionalFilter")
                || self.has("RWeightedFilter")
                || self.has("MtfFile")
                || self.has("StockCurve"))
        {
            return Err("dose weighting conflicts with selected filter mode".into());
        }
        Ok(())
    }
}

impl MtfFilterOptions {
    pub fn new(ctf: Vec<f32>, delta: f32) -> Result<Self, String> {
        if ctf.is_empty() || delta <= 0.0 {
            return Err("invalid MTF filter table".into());
        }
        Ok(Self {
            ctf,
            delta,
            one_dimensional: false,
            density_scale: 1.0,
        })
    }
}

pub fn fft_filter_3d(
    array: &mut [f32],
    nx_dim: usize,
    ny: usize,
    nz: usize,
    ctf: &[f32],
    delta: f32,
) -> Result<(), String> {
    if nx_dim < 2 || ny == 0 || nz == 0 || delta <= 0.0 || array.len() != 2 * nx_dim * ny * nz {
        return Err("invalid 3-D FFT filter dimensions".into());
    }
    let del_x = 0.5 / (nx_dim as f32 - 1.0);
    let del_y = 1.0 / ny as f32;
    let del_z = 1.0 / nz as f32;
    for z in 0..nz {
        let mut za = z as f32 * del_z;
        if za > 0.5 {
            za = 1.0 - za;
        }
        for y in 0..ny {
            let mut ya = y as f32 * del_y;
            if ya > 0.5 {
                ya = 1.0 - ya;
            }
            for x in 0..nx_dim {
                let xa = x as f32 * del_x;
                let index =
                    ((xa.mul_add(xa, ya.mul_add(ya, za * za))).sqrt() / delta + 0.5) as usize;
                let filter = *ctf.get(index).ok_or("CTF table is too short")?;
                let base = (z * ny + y) * nx_dim * 2 + x * 2;
                array[base] *= filter;
                array[base + 1] *= filter;
            }
        }
    }
    Ok(())
}

pub fn fft_filter_1d(
    array: &mut [f32],
    nx_dim: usize,
    ny: usize,
    ctf: &[f32],
    delta: f32,
) -> Result<(), String> {
    if nx_dim < 2 || ny == 0 || delta <= 0.0 || array.len() != 2 * nx_dim * ny {
        return Err("invalid 1-D FFT filter dimensions".into());
    }
    let del_x = 0.5 / (nx_dim as f32 - 1.0);
    for y in 0..ny {
        for x in 0..nx_dim {
            let index = (x as f32 * del_x / delta + 0.5) as usize;
            let filter = *ctf.get(index).ok_or("CTF table is too short")?;
            let base = (y * nx_dim + x) * 2;
            array[base] *= filter;
            array[base + 1] *= filter;
        }
    }
    Ok(())
}

/// Source `amplifier`, returning its table and updated `(delta, nsize)`.
pub fn amplifier(
    cutoff: f32,
    amplitude: f32,
    power: f32,
    nx: usize,
    ny: usize,
) -> Result<(Vec<f32>, f32), String> {
    if cutoff <= 0.0 || nx == 0 || ny == 0 {
        return Err("invalid amplifier parameters".into());
    }
    let nsize = (2 * nx).max(2 * ny).clamp(1024, 8192);
    let delta = 1.0 / (0.71 * nsize as f32);
    let ctf = (0..nsize)
        .map(|index| {
            ((1.0 + (amplitude - 1.0) * (-(index as f32 * delta / cutoff).powf(power)).exp())
                / amplitude)
                .max(1.0e-6)
        })
        .collect();
    Ok((ctf, delta))
}

pub fn adjusted_fake_iter(nominal: f32) -> f32 {
    if nominal > 30.0 {
        27.0 + 0.6 * (nominal - 30.0)
    } else if nominal > 15.0 {
        15.0 + 0.8 * (nominal - 15.0)
    } else {
        nominal
    }
}

pub fn deconv_filter(
    deconv_strength: f32,
    snr_falloff: f32,
    high_pass_nyq: f32,
    voltage: f32,
    cs: f32,
    defocus: f32,
    phase_shift: f32,
    ang_pix_size: f32,
    ctf: &mut [f32],
    delta: f32,
) -> Result<(), String> {
    if voltage <= 0.0 || ang_pix_size <= 0.0 || delta <= 0.0 || ctf.is_empty() {
        return Err("invalid deconvolution parameters".into());
    }
    let pixel_size_m = ang_pix_size * 1.0e-10;
    let pi = 3.141593_f32;
    let lambda = 12.2643247 / (voltage * (1.0 + voltage * 0.978466e-6)).sqrt() * 1.0e-10;
    ctf[0] = 1.0;
    for (index, value) in ctf.iter_mut().enumerate().skip(1) {
        let fraction_nyquist = 2.0 * index as f32 * delta;
        let high_pass = if high_pass_nyq > 0.0 {
            1.0 - (fraction_nyquist.div_euclid(high_pass_nyq).min(1.0) * pi).cos()
        } else {
            2.0
        };
        let snr = (-fraction_nyquist * snr_falloff * 100.0 / ang_pix_size).exp()
            * 10_f32.powf(3.0 * deconv_strength)
            * high_pass
            + 1.0e-6;
        let k = fraction_nyquist / (2.0 * pixel_size_m);
        let k2 = k * k;
        let w =
            pi / 2.0 * (lambda.powi(3) * cs * k2 * k2 + 2.0 * lambda * defocus * k2) - phase_shift;
        let contrast_transfer = w.cos() * 0.07 - (1.0 - 0.07_f32.powi(2)).sqrt() * w.sin();
        *value = contrast_transfer.abs() / (contrast_transfer.powi(2) + 1.0 / snr);
    }
    Ok(())
}

/// Source `convertFreqUnit`, returning frequency in inverse pixels.
pub fn convert_freq_unit(inv_freq_unit: i32, pixel_size_nm: f32, frequency: f32) -> f32 {
    if inv_freq_unit == 0 {
        return frequency;
    }
    let mut result = if inv_freq_unit > 0 && frequency > 0.0 {
        frequency.recip()
    } else {
        frequency
    };
    result *= pixel_size_nm;
    if inv_freq_unit.unsigned_abs() % 2 == 0 {
        result *= 10.0;
    }
    result
}

/// Safe 2-D equivalent of the ordinary per-section branch in `main`.
/// It uses IMOD's RustFFT packed `(nx + 2) * ny` layout, applies the source
/// radial/one-dimensional filter, then restores an owned float MRC volume.
pub fn filter_real_mrc_slices(input: &IoMrc, options: &MtfFilterOptions) -> Result<IoMrc, String> {
    let nx = usize::try_from(input.header.nx).map_err(|_| "negative MRC width")?;
    let ny = usize::try_from(input.header.ny).map_err(|_| "negative MRC height")?;
    let nz = usize::try_from(input.header.nz).map_err(|_| "negative MRC depth")?;
    if nx == 0 || ny == 0 || nz == 0 || nx % 2 != 0 {
        return Err("MTF 2-D filtering requires nonzero even dimensions".into());
    }
    let mut result = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        let mut packed = vec![0.0; (nx + 2) * ny];
        let mut slice = vec![0.0; nx * ny];
        if !input.read_mrc_slice_float(z as i32, &mut slice) {
            return Err("cannot read MRC scalar slice".into());
        }
        for (row, source) in packed.chunks_exact_mut(nx + 2).zip(slice.chunks_exact(nx)) {
            row[..nx].copy_from_slice(source);
        }
        crate::imod::libfft::rustfft_backend::todfft(&mut packed, nx as i32, ny as i32, 0)?;
        if options.one_dimensional {
            fft_filter_1d(&mut packed, nx / 2 + 1, ny, &options.ctf, options.delta)?;
        } else {
            fft_filter_3d(&mut packed, nx / 2 + 1, ny, 1, &options.ctf, options.delta)?;
        }
        crate::imod::libfft::rustfft_backend::todfft(&mut packed, nx as i32, ny as i32, 1)?;
        for row in packed.chunks_exact(nx + 2) {
            result.extend(row[..nx].iter().map(|value| value * options.density_scale));
        }
    }
    let mut output = IoMrc::new(MrcVolume::Float(result), nx as i32, ny as i32, nz as i32, 2);
    output.header = input.header.clone();
    output.header.mode = 2;
    output.create_header();
    Ok(output)
}

pub fn filter_packed_fft_mrc(input: &IoMrc, options: &MtfFilterOptions) -> Result<IoMrc, String> {
    let nx_dim = usize::try_from(input.header.nx).map_err(|_| "negative FFT width")?;
    let ny = usize::try_from(input.header.ny).map_err(|_| "negative FFT height")?;
    let nz = usize::try_from(input.header.nz).map_err(|_| "negative FFT depth")?;
    let MrcVolume::ComplexFloat(samples) = input.volume.as_ref().ok_or("MRC has no volume")? else {
        return Err("packed FFT filtering requires complex-float MRC data".into());
    };
    let mut packed = samples.iter().flat_map(|pair| *pair).collect::<Vec<_>>();
    fft_filter_3d(&mut packed, nx_dim, ny, nz, &options.ctf, options.delta)?;
    let samples = packed
        .chunks_exact(2)
        .map(|pair| [pair[0], pair[1]])
        .collect();
    let mut output = IoMrc::new(
        MrcVolume::ComplexFloat(samples),
        nx_dim as i32,
        ny as i32,
        nz as i32,
        4,
    );
    output.header = input.header.clone();
    Ok(output)
}

pub fn filter_real_mrc_3d(input: &IoMrc, options: &MtfFilterOptions) -> Result<IoMrc, String> {
    let nx = usize::try_from(input.header.nx).map_err(|_| "negative MRC width")?;
    let ny = usize::try_from(input.header.ny).map_err(|_| "negative MRC height")?;
    let nz = usize::try_from(input.header.nz).map_err(|_| "negative MRC depth")?;
    if nx == 0 || ny == 0 || nz == 0 || nx % 2 != 0 {
        return Err("3-D MTF filtering requires nonzero even dimensions".into());
    }
    let stride = nx + 2;
    let mut packed = vec![0.0; stride * ny * nz];
    for z in 0..nz {
        let mut slice = vec![0.0; nx * ny];
        if !input.read_mrc_slice_float(z as i32, &mut slice) {
            return Err("cannot read MRC scalar slice".into());
        }
        for (target, source) in packed[z * stride * ny..(z + 1) * stride * ny]
            .chunks_exact_mut(stride)
            .zip(slice.chunks_exact(nx))
        {
            target[..nx].copy_from_slice(source);
        }
    }
    let mut workspace = vec![0.0; 2 * nz * ((nx + 2) / 2)];
    crate::imod::libfft::thrdfft::thrdfft(
        &mut packed,
        &mut workspace,
        nx as i32,
        ny as i32,
        nz as i32,
        0,
    );
    fft_filter_3d(&mut packed, nx / 2 + 1, ny, nz, &options.ctf, options.delta)?;
    crate::imod::libfft::thrdfft::thrdfft(
        &mut packed,
        &mut workspace,
        nx as i32,
        ny as i32,
        nz as i32,
        -1,
    );
    let values = packed
        .chunks_exact(stride)
        .flat_map(|row| row[..nx].iter().map(|value| value * options.density_scale))
        .collect();
    let mut output = IoMrc::new(MrcVolume::Float(values), nx as i32, ny as i32, nz as i32, 2);
    output.header = input.header.clone();
    output.header.mode = 2;
    output.create_header();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_filters_preserve_complex_pair_layout() {
        let mut values = vec![2., 3., 4., 5., 6., 7., 8., 9.];
        fft_filter_1d(&mut values, 2, 2, &[0.5, 0.25], 0.5).unwrap();
        assert_eq!(values, vec![1., 1.5, 1., 1.25, 3., 3.5, 2., 2.25]);
    }
    #[test]
    fn frequency_and_fake_iteration_follow_source_branches() {
        assert_eq!(adjusted_fake_iter(40.), 33.);
        assert!((convert_freq_unit(2, 2., 5.) - 4.).abs() < 1.0e-6);
    }
    #[test]
    fn packed_mrc_path_filters_real_and_imaginary_components() {
        let input = IoMrc::new(
            MrcVolume::ComplexFloat(vec![[2., 3.], [4., 5.]]),
            2,
            1,
            1,
            4,
        );
        let options = MtfFilterOptions::new(vec![0.5, 0.25], 0.5).unwrap();
        let output = filter_packed_fft_mrc(&input, &options).unwrap();
        assert_eq!(
            output.volume,
            Some(MrcVolume::ComplexFloat(vec![[1., 1.5], [1., 1.25]]))
        );
    }
    #[test]
    fn source_aliases_and_dose_conflicts_are_parsed() {
        let arguments = [
            "-input", "in.mrc", "-output", "out.mrc", "-zrange", "1", "3", "-noise",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        let command = MtfFilterCommand::parse(&arguments).unwrap();
        assert_eq!(command.values("StartingAndEndingZ").unwrap(), ["1", "3"]);
        let bad = ["-input", "in.mrc", "-dfixed", "1", "-3dfilter"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(MtfFilterCommand::parse(&bad).is_err());
    }
}
