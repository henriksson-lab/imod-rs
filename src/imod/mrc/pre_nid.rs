//! Owned computational translation of `IMOD/mrc/preNID.cpp`.
//!
//! The original program's `float **` matrices and `matrix/free_matrix` calls
//! are represented by one contiguous image type.  Recursive filtering uses
//! the translated Recline implementation, not an FFI boundary.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_BYTE, MRC_MODE_FLOAT, MRC_MODE_SHORT, MrcHeader, mrc_head_label, mrc_head_read,
    mrc_head_write, mrc_init_output_header, mrc_read_float_slice, mrc_write_slice,
};
use crate::imod::mrc::recline::{
    DerivativeOrder, RecursiveFilterType, init_recursive_coefficients, recursive_filter_1d,
};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct PreNidImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}
impl PreNidImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[x * self.height + y]
    }
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        self.data[x * self.height + y] = value;
    }
}

#[derive(Clone, Debug)]
pub struct PreNidParameters {
    pub sigma: Vec<f32>,
    pub alpha: Vec<f32>,
    pub beta: Vec<f32>,
    pub tau: Vec<f32>,
    pub iterations: Vec<usize>,
    pub padding: usize,
    pub spacing_x: f64,
    pub spacing_y: f64,
    pub mask_output: bool,
}
impl PreNidParameters {
    pub fn new(sigma: Vec<f32>) -> Result<Self, String> {
        if sigma.is_empty() {
            return Err("no sigma is specified".into());
        }
        let count = sigma.len();
        Ok(Self {
            sigma,
            alpha: vec![0.5; count],
            beta: vec![0.5; count],
            tau: vec![0.1; count],
            iterations: vec![1; count],
            padding: 40,
            spacing_x: 1.0,
            spacing_y: 1.0,
            mask_output: false,
        })
    }
    fn validate(&self) -> Result<(), String> {
        let n = self.sigma.len();
        if n == 0
            || self.alpha.len() != n
            || self.beta.len() != n
            || self.tau.len() != n
            || self.iterations.len() != n
        {
            Err("sigma, alpha, beta, tau, and iterations must have equal nonzero lengths".into())
        } else {
            Ok(())
        }
    }
}

/// Source `separateStringCommaValues`.
pub fn separate_string_comma_values(input: &str) -> Vec<f32> {
    input
        .split(',')
        .map(|value| value.trim().parse().unwrap_or(0.0))
        .collect()
}
/// Source `FillingPadding` for an owned padded image.
pub fn filling_padding(image: &mut PreNidImage, nx: usize, ny: usize, padding: usize) {
    if nx <= padding + 1 || ny <= padding + 1 {
        return;
    }
    for p in 0..padding {
        for x in 0..nx + 2 * padding {
            let bottom = image.get(x, ny - 1 - p);
            let top = image.get(x, 2 * padding - p);
            image.set(x, padding + ny + p, bottom);
            image.set(x, p, top);
        }
        for y in 0..ny + 2 * padding {
            let right = image.get(nx - 1 - p, y);
            let left = image.get(2 * padding - p, y);
            image.set(padding + nx + p, y, right);
            image.set(p, y, left);
        }
    }
}
/// Source `gaussRecursiveDerivatives1D`.
pub fn gauss_recursive_derivatives_1d(
    sigma: f64,
    nx: usize,
    ny: usize,
    padding: usize,
    direction: usize,
    derivative: usize,
    input: &mut PreNidImage,
    output: &mut PreNidImage,
) -> Result<(), String> {
    filling_padding(input, nx, ny, padding);
    if sigma < 0.1 {
        output.data.clone_from(&input.data);
        filling_padding(output, nx, ny, padding);
        return Ok(());
    }
    let derivative = match derivative {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::One,
        2 => DerivativeOrder::Two,
        3 => DerivativeOrder::Three,
        _ => return Err("invalid derivative order".into()),
    };
    let coefficients =
        init_recursive_coefficients(sigma, RecursiveFilterType::GaussianDeriche, derivative)?;
    if direction == 0 {
        for y in 0..ny + 2 * padding {
            let line: Vec<_> = (0..nx + 2 * padding)
                .map(|x| input.get(x, y) as f64)
                .collect();
            let filtered = recursive_filter_1d(&coefficients, &line)?;
            for (x, value) in filtered.into_iter().enumerate() {
                output.set(x, y, value as f32);
            }
        }
    } else if direction == 1 {
        for x in 0..nx + 2 * padding {
            let line: Vec<_> = (0..ny + 2 * padding)
                .map(|y| input.get(x, y) as f64)
                .collect();
            let filtered = recursive_filter_1d(&coefficients, &line)?;
            for (y, value) in filtered.into_iter().enumerate() {
                output.set(x, y, value as f32);
            }
        }
    } else {
        return Err("invalid filter direction".into());
    }
    filling_padding(output, nx, ny, padding);
    Ok(())
}
/// Source `CreateMaskedLocalSmooth`.
pub fn create_masked_local_smooth(
    nx: usize,
    ny: usize,
    padding: usize,
    sigma: f64,
    alpha: f64,
    beta: f64,
    tau: f64,
    image: &mut PreNidImage,
    mask_out: &mut PreNidImage,
    graded: &mut PreNidImage,
    smoothed: &mut PreNidImage,
    normalized_dog: &mut PreNidImage,
) -> Result<f64, String> {
    let size = (nx + 2 * padding, ny + 2 * padding);
    let mut blur = PreNidImage::new(size.0, size.1);
    let mut blur_a = PreNidImage::new(size.0, size.1);
    let mut dog = PreNidImage::new(size.0, size.1);
    let mut dx = PreNidImage::new(size.0, size.1);
    let mut dy = PreNidImage::new(size.0, size.1);
    gauss_recursive_derivatives_1d(sigma, nx, ny, padding, 0, 0, image, &mut blur)?;
    gauss_recursive_derivatives_1d(sigma, nx, ny, padding, 1, 0, &mut blur.clone(), &mut blur)?;
    for i in 0..image.data.len() {
        dog.data[i] = image.data[i] - blur.data[i];
    }
    gauss_recursive_derivatives_1d(
        sigma * alpha,
        nx,
        ny,
        padding,
        0,
        0,
        &mut dog.clone(),
        &mut dog,
    )?;
    gauss_recursive_derivatives_1d(
        sigma * alpha,
        nx,
        ny,
        padding,
        1,
        0,
        &mut dog.clone(),
        &mut dog,
    )?;
    let mut max_negative = 0.0f32;
    for x in padding..nx + padding {
        for y in padding..ny + padding {
            let value = (-dog.get(x, y)).max(0.0);
            normalized_dog.set(x, y, value);
            max_negative = max_negative.max(value);
        }
    }
    if max_negative > 0.0 {
        for value in &mut normalized_dog.data {
            *value /= max_negative;
        }
    }
    gauss_recursive_derivatives_1d(
        sigma * alpha,
        nx,
        ny,
        padding,
        0,
        1,
        &mut dog.clone(),
        &mut dx,
    )?;
    gauss_recursive_derivatives_1d(
        sigma * alpha,
        nx,
        ny,
        padding,
        1,
        1,
        &mut dog.clone(),
        &mut dy,
    )?;
    for i in 0..dog.data.len() {
        dog.data[i] = dx.data[i].powi(2) + dy.data[i].powi(2);
    }
    gauss_recursive_derivatives_1d(
        sigma * alpha,
        nx,
        ny,
        padding,
        0,
        0,
        &mut dog.clone(),
        &mut dog,
    )?;
    gauss_recursive_derivatives_1d(
        sigma * alpha,
        nx,
        ny,
        padding,
        1,
        0,
        &mut dog.clone(),
        &mut dog,
    )?;
    let mut relevant = Vec::new();
    for x in padding..nx + padding {
        for y in padding..ny + padding {
            if normalized_dog.get(x, y) > 0.0 {
                relevant.push(dog.get(x, y));
            }
        }
    }
    let mean = relevant.iter().map(|&v| v as f64).sum::<f64>() / (relevant.len().max(1) as f64);
    let min = relevant.iter().copied().fold(f32::INFINITY, f32::min);
    let max = relevant.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(f32::EPSILON);
    gauss_recursive_derivatives_1d(beta, nx, ny, padding, 0, 0, image, &mut blur_a)?;
    gauss_recursive_derivatives_1d(
        beta,
        nx,
        ny,
        padding,
        1,
        0,
        &mut blur_a.clone(),
        &mut blur_a,
    )?;
    for x in 0..size.0 {
        for y in 0..size.1 {
            let irregular = ((dog.get(x, y) - mean as f32) / range).max(0.0);
            let selected = normalized_dog.get(x, y) > 0.01 && irregular > tau as f32;
            mask_out.set(x, y, if selected { 1.0 } else { 0.0 });
            smoothed.set(
                x,
                y,
                if selected {
                    blur_a.get(x, y)
                } else {
                    image.get(x, y)
                },
            );
            graded.set(x, y, irregular);
        }
    }
    Ok(0.0)
}
/// Source `LinearityEnhancingDiffusion` (the source's active eight-neighbor evolution).
pub fn linearity_enhancing_diffusion(
    nx: usize,
    ny: usize,
    padding: usize,
    spacing_x: f64,
    spacing_y: f64,
    input: &PreNidImage,
    output: &mut PreNidImage,
    mask: &PreNidImage,
) {
    let rxx = (0.1 / (spacing_x * spacing_x)) as f32;
    let ryy = (0.1 / (spacing_y * spacing_y)) as f32;
    let rxy = (0.1 / (1.4142 * spacing_x * spacing_y)) as f32;
    for x in padding..nx + padding {
        for y in padding..ny + padding {
            let here = input.get(x, y);
            let w = mask.get(x, y);
            output.set(
                x,
                y,
                here + w
                    * (rxx * (input.get(x + 1, y) - here)
                        + rxx * (input.get(x - 1, y) - here)
                        + ryy * (input.get(x, y + 1) - here)
                        + ryy * (input.get(x, y - 1) - here)
                        + rxy * (input.get(x + 1, y + 1) - here)
                        + rxy * (input.get(x - 1, y - 1) - here)
                        + rxy * (input.get(x - 1, y + 1) - here)
                        + rxy * (input.get(x + 1, y - 1) - here)),
            );
        }
    }
}
/// Source `automaticPreLR`, using `Vec<PreNidImage>` in place of three raw stack arrays.
pub fn automatic_pre_lr(
    stack: &mut [PreNidImage],
    parameters: &PreNidParameters,
    input_masks: Option<&[PreNidImage]>,
) -> Result<Vec<PreNidImage>, String> {
    parameters.validate()?;
    let mut output = stack.to_vec();
    for (section, image) in stack.iter_mut().enumerate() {
        let nx = image.width - 2 * parameters.padding;
        let ny = image.height - 2 * parameters.padding;
        let mut current = image.clone();
        let mut mask = PreNidImage::new(image.width, image.height);
        let mut graded = mask.clone();
        let mut smooth = mask.clone();
        let mut dog = mask.clone();
        for step in 0..parameters.sigma.len() {
            if let Some(masks) = input_masks {
                mask = masks[section].clone();
            } else {
                create_masked_local_smooth(
                    nx,
                    ny,
                    parameters.padding,
                    parameters.sigma[step] as f64,
                    parameters.alpha[step] as f64,
                    parameters.beta[step] as f64,
                    parameters.tau[step] as f64,
                    &mut current,
                    &mut mask,
                    &mut graded,
                    &mut smooth,
                    &mut dog,
                )?;
            }
            for _ in 0..parameters.iterations[step] {
                let mut next = current.clone();
                linearity_enhancing_diffusion(
                    nx,
                    ny,
                    parameters.padding,
                    parameters.spacing_x,
                    parameters.spacing_y,
                    &current,
                    &mut next,
                    &mask,
                );
                current = next;
            }
        }
        output[section] = if parameters.mask_output {
            let mut shown = mask;
            for value in &mut shown.data {
                *value *= 100.0;
            }
            shown
        } else {
            current
        };
    }
    Ok(output)
}

/// File-level core of source `main`, excluding only Pip's presentation layer.
/// It accepts already-parsed options, reads real MRC slices as floats, applies
/// the complete owned filter stack, and preserves the input output mode.
pub fn run_pre_nid(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    parameters: &PreNidParameters,
    input_mask: Option<impl AsRef<Path>>,
) -> Result<(), String> {
    parameters.validate()?;
    let mut source = ImodFile::open(input.as_ref(), "rb").ok_or("could not open input file")?;
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut source, &mut header) != 0 {
        return Err("reading header of input file".into());
    }
    if !matches!(header.mode, MRC_MODE_BYTE | MRC_MODE_SHORT | MRC_MODE_FLOAT) {
        return Err(format!(
            "file mode is {}; only byte, short integer, or real allowed",
            header.mode
        ));
    }
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let nz = header.nz as usize;
    let full_x = nx + 2 * parameters.padding;
    let full_y = ny + 2 * parameters.padding;
    let mut stack = Vec::with_capacity(nz);
    for z in 0..nz {
        let mut values = vec![0.0; nx * ny];
        if mrc_read_float_slice(&mut values, &mut header, z as i32) != 0 {
            return Err(format!("reading slice {z}"));
        }
        let mut image = PreNidImage::new(full_x, full_y);
        for x in 0..nx {
            for y in 0..ny {
                image.set(
                    x + parameters.padding,
                    y + parameters.padding,
                    values[x + y * nx],
                );
            }
        }
        filling_padding(&mut image, nx, ny, parameters.padding);
        stack.push(image);
    }
    let masks = if let Some(path) = input_mask {
        let mut file = ImodFile::open(path.as_ref(), "rb").ok_or("could not open mask file")?;
        let mut mask_header = MrcHeader::default();
        if mrc_head_read(&mut file, &mut mask_header) != 0
            || (mask_header.nx, mask_header.ny, mask_header.nz) != (header.nx, header.ny, header.nz)
        {
            return Err("mask size not compatible with input image size".into());
        }
        let mut masks = Vec::with_capacity(nz);
        for z in 0..nz {
            let mut values = vec![0.0; nx * ny];
            if mrc_read_float_slice(&mut values, &mut mask_header, z as i32) != 0 {
                return Err(format!("reading mask slice {z}"));
            }
            let mut image = PreNidImage::new(full_x, full_y);
            for x in 0..nx {
                for y in 0..ny {
                    image.set(
                        x + parameters.padding,
                        y + parameters.padding,
                        if values[x + y * nx] > 0.000001 {
                            1.0
                        } else {
                            0.0
                        },
                    );
                }
            }
            masks.push(image);
        }
        Some(masks)
    } else {
        None
    };
    let result = automatic_pre_lr(&mut stack, parameters, masks.as_deref())?;
    let mut destination = ImodFile::open(output, "wb").ok_or("could not open output file")?;
    let mut out_header = header.clone();
    mrc_init_output_header(&mut out_header);
    mrc_head_label(&mut out_header, b"PreNID filtered image");
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut mean = 0.0f64;
    for (z, image) in result.iter().enumerate() {
        let mut pixels = Vec::with_capacity(nx * ny);
        for y in 0..ny {
            for x in 0..nx {
                pixels.push(image.get(x + parameters.padding, y + parameters.padding));
            }
        }
        for value in &pixels {
            min = min.min(*value);
            max = max.max(*value);
            mean += *value as f64 / (nx * ny * nz) as f64;
        }
        let bytes = match out_header.mode {
            MRC_MODE_FLOAT => pixels
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
            MRC_MODE_SHORT => pixels
                .iter()
                .flat_map(|v| (*v as i16).to_ne_bytes())
                .collect::<Vec<_>>(),
            MRC_MODE_BYTE => pixels.iter().map(|v| *v as u8).collect(),
            _ => return Err("unsupported output mode".into()),
        };
        if mrc_write_slice(&bytes, &mut destination, &mut out_header, z as i32, b'Z') != 0 {
            return Err(format!("writing slice {z}"));
        }
    }
    out_header.amin = min;
    out_header.amax = max;
    out_header.amean = mean as f32;
    if mrc_head_write(&mut destination, &mut out_header) != 0 {
        return Err("writing output header".into());
    }
    Ok(())
}

/// Parsed command state for the C `main` Pip options.
#[derive(Clone, Debug)]
pub struct PreNidCommand {
    pub input: std::path::PathBuf,
    pub output: std::path::PathBuf,
    pub angles: Option<std::path::PathBuf>,
    pub input_mask: Option<std::path::PathBuf>,
    pub views: Option<Vec<usize>>,
    pub parameters: PreNidParameters,
}

fn values_option(
    values: Option<&String>,
    count: usize,
    default: f32,
    name: &str,
) -> Result<Vec<f32>, String> {
    let values = match values {
        Some(text) => separate_string_comma_values(text),
        None => vec![default; count],
    };
    if values.len() != count {
        Err(format!(
            "Error with '--{name}' option: be sure it is consistent with the '--Sigma' option."
        ))
    } else {
        Ok(values)
    }
}

/// Pip-equivalent command parser for source `main`.  Long field names and the
/// autodoc short forms are both accepted.  `ViewsToProcess` is retained in
/// parsed state exactly like the source's currently-unused list.
pub fn parse_pre_nid(arguments: &[String]) -> Result<PreNidCommand, String> {
    let mut input = None;
    let mut output = None;
    let mut angles = None;
    let mut input_mask = None;
    let mut sigma = None;
    let mut alpha = None;
    let mut beta = None;
    let mut tau = None;
    let mut iterations = None;
    let mut views = None;
    let mut mask_output = false;
    let mut index = 1usize;
    while index < arguments.len() {
        let option = &arguments[index];
        let take = |index: &mut usize| -> Result<String, String> {
            *index += 1;
            arguments
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("missing value for {option}"))
        };
        match option.as_str() {
            "-input" | "--input" | "-InputStack" | "--InputStack" => {
                input = Some(take(&mut index)?)
            }
            "-output" | "--output" | "-OutputFileName" | "--OutputFileName" => {
                output = Some(take(&mut index)?)
            }
            "-angles" | "--angles" | "-AnglesFile" | "--AnglesFile" => {
                angles = Some(take(&mut index)?)
            }
            "-im" | "--im" | "-InputMask" | "--InputMask" => input_mask = Some(take(&mut index)?),
            "-s" | "--s" | "-Sigma" | "--Sigma" => sigma = Some(take(&mut index)?),
            "-a" | "--a" | "-Alpha" | "--Alpha" => alpha = Some(take(&mut index)?),
            "-b" | "--b" | "-Beta" | "--Beta" => beta = Some(take(&mut index)?),
            "-t" | "--t" | "-Tau" | "--Tau" => tau = Some(take(&mut index)?),
            "-ite" | "--ite" | "-Iterations" | "--Iterations" => {
                iterations = Some(take(&mut index)?)
            }
            "-views" | "--views" | "-ViewsToProcess" | "--ViewsToProcess" => {
                views = Some(take(&mut index)?)
            }
            "-mask" | "--mask" | "-MaskOutput" | "--MaskOutput" => mask_output = true,
            "-help" | "--help" => {
                return Err(
                    "usage: preNID -input stack -output stack -sigma values [options]".into(),
                );
            }
            value if value.starts_with('-') => return Err(format!("unknown option {value}")),
            value => return Err(format!("unexpected non-option argument {value}")),
        };
        index += 1;
    }
    let sigma = separate_string_comma_values(&sigma.ok_or("No sigma is specified, aborting...")?);
    if sigma.is_empty() {
        return Err("No sigma is specified, aborting...".into());
    }
    let mut parameters = PreNidParameters::new(sigma.clone())?;
    parameters.alpha = values_option(alpha.as_ref(), sigma.len(), 0.5, "Alpha")?;
    parameters.beta = values_option(beta.as_ref(), sigma.len(), 0.5, "Beta")?;
    parameters.tau = values_option(tau.as_ref(), sigma.len(), 0.1, "Tau")?;
    parameters.iterations = values_option(iterations.as_ref(), sigma.len(), 1.0, "Iterations")?
        .into_iter()
        .map(|value| value as usize)
        .collect();
    parameters.mask_output = mask_output;
    let parsed_views = views.map(|text| {
        text.split(',')
            .flat_map(|part| {
                if let Some((first, last)) = part.split_once('-') {
                    let first = first.trim().parse::<usize>().ok();
                    let last = last.trim().parse::<usize>().ok();
                    first
                        .into_iter()
                        .flat_map(move |first| last.into_iter().flat_map(move |last| first..=last))
                        .collect::<Vec<_>>()
                } else {
                    part.trim().parse().ok().into_iter().collect()
                }
            })
            .collect()
    });
    Ok(PreNidCommand {
        input: input.map(Into::into).ok_or("No stack specified")?,
        output: output
            .map(Into::into)
            .ok_or("OutputFileName is not specified")?,
        angles: angles.map(Into::into),
        input_mask: input_mask.map(Into::into),
        views: parsed_views,
        parameters,
    })
}

/// C `main` after Pip parsing.  Angle data is parsed by the source only to
/// populate currently unused metadata; the computational path is identical.
pub fn pre_nid(arguments: &[String]) -> Result<(), String> {
    let command = parse_pre_nid(arguments)?;
    run_pre_nid(
        command.input,
        command.output,
        &command.parameters,
        command.input_mask,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn comma_parser_and_padding_match_source() {
        assert_eq!(separate_string_comma_values("1, 2.5,x"), vec![1., 2.5, 0.]);
        let mut image = PreNidImage::new(6, 6);
        image.set(2, 2, 7.);
        filling_padding(&mut image, 2, 2, 2);
        assert_eq!(image.get(0, 2), 7.);
    }
    #[test]
    fn diffusion_is_owned_and_masks_only_selected_pixels() {
        let mut image = PreNidImage::new(7, 7);
        image.set(3, 3, 10.);
        let mut mask = PreNidImage::new(7, 7);
        mask.set(3, 3, 1.);
        let mut out = image.clone();
        linearity_enhancing_diffusion(3, 3, 2, 1., 1., &image, &mut out, &mask);
        assert!(out.get(3, 3) < 10.);
    }
    #[test]
    fn cli_maps_source_autodoc_names_and_defaults() {
        let command = parse_pre_nid(&[
            "preNID".into(),
            "-input".into(),
            "in.mrc".into(),
            "-output".into(),
            "out.mrc".into(),
            "-s".into(),
            "2,3".into(),
            "-views".into(),
            "1-2,4".into(),
            "-mask".into(),
        ])
        .unwrap();
        assert_eq!(command.parameters.alpha, vec![0.5, 0.5]);
        assert!(command.parameters.mask_output);
        assert_eq!(command.views, Some(vec![1, 2, 4]));
    }
    #[test]
    fn cli_requires_sigma_and_parallel_lists() {
        assert!(parse_pre_nid(&["preNID".into()]).is_err());
        assert!(
            parse_pre_nid(&[
                "preNID".into(),
                "-input".into(),
                "a".into(),
                "-output".into(),
                "b".into(),
                "-s".into(),
                "1,2".into(),
                "-a".into(),
                "1".into()
            ])
            .is_err()
        );
    }
}
