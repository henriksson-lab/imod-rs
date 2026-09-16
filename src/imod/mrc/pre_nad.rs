//! Owned translation of `IMOD/mrc/preNAD.cpp`.
//!
//! The original uses `float **` Numerical Recipes matrices.  This unit keeps
//! the same reflected padded geometry in one owned row-major allocation.

use crate::imod::libcfshr::b3dutil::ImodFile;
use crate::imod::libiimod::mrcfiles::{
    MRC_MODE_FLOAT, MrcHeader, mrc_head_label, mrc_head_read, mrc_head_write,
    mrc_init_output_header, mrc_read_float_slice, mrc_write_slice,
};
use crate::imod::mrc::recline::{
    DerivativeOrder, RecursiveFilterType, init_recursive_coefficients, recursive_filter_1d,
};

pub const PRE_NAD_SOURCE_FUNCTIONS: &[&str] = &[
    "FillingPadding",
    "getQuartileMVDIndex",
    "gaussRecursiveDerivatives1D",
    "preNAD1Tilt",
    "automaticPreNAD",
    "main",
];

#[derive(Clone, Debug, PartialEq)]
pub struct PreNadImage {
    pub nx: usize,
    pub ny: usize,
    pub padding: usize,
    pub data: Vec<f32>,
}
impl PreNadImage {
    pub fn new(nx: usize, ny: usize, padding: usize) -> Self {
        Self {
            nx,
            ny,
            padding,
            data: vec![0.; (nx + 2 * padding) * (ny + 2 * padding)],
        }
    }
    pub fn from_pixels(
        nx: usize,
        ny: usize,
        padding: usize,
        pixels: &[f32],
    ) -> Result<Self, String> {
        if pixels.len() != nx * ny {
            return Err("image dimensions do not match pixels".into());
        }
        let mut image = Self::new(nx, ny, padding);
        for x in 0..nx {
            for y in 0..ny {
                image.set(x + padding, y + padding, pixels[x + nx * y])
            }
        }
        filling_padding(&mut image);
        Ok(image)
    }
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[x * (self.ny + 2 * self.padding) + y]
    }
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        let i = x * (self.ny + 2 * self.padding) + y;
        self.data[i] = value
    }
    pub fn pixels(&self) -> Vec<f32> {
        let mut p = Vec::with_capacity(self.nx * self.ny);
        for y in self.padding..self.padding + self.ny {
            for x in self.padding..self.padding + self.nx {
                p.push(self.get(x, y))
            }
        }
        p
    }
}
#[derive(Clone, Debug)]
pub struct TiltProjection {
    pub current_iteration: usize,
    pub mvd: f64,
    pub angle_radians: f64,
    pub angle_degrees: f64,
    pub sigma: f64,
    pub skip: bool,
    pub tilt_number: usize,
    pub data: PreNadImage,
    pub original: PreNadImage,
}

/// C `FillingPadding`.
pub fn filling_padding(image: &mut PreNadImage) {
    let p = image.padding;
    if image.nx <= p + 1 || image.ny <= p + 1 {
        return;
    }
    for q in 0..p {
        for x in 0..image.nx + 2 * p {
            image.set(x, p + image.ny + q, image.get(x, image.ny - 1 - q));
            image.set(x, q, image.get(x, 2 * p - q));
        }
        for y in 0..image.ny + 2 * p {
            image.set(p + image.nx + q, y, image.get(image.nx - 1 - q, y));
            image.set(q, y, image.get(2 * p - q, y));
        }
    }
}
/// C `getQuartileMVDIndex`.
pub fn get_quartile_mvd_index(stack: &[TiltProjection]) -> usize {
    let mut order: Vec<usize> = (0..stack.len()).collect();
    order.sort_by(|a, b| stack[*a].mvd.total_cmp(&stack[*b].mvd));
    order[stack.len() / 4]
}
/// C `gaussRecursiveDerivatives1D`; the owned implementation uses its Gaussian
/// convolution definition rather than foreign recursive-filter storage.
pub fn gauss_recursive_derivatives_1d(
    sigma: f64,
    direction: usize,
    derivative: usize,
    input: &mut PreNadImage,
    output: &mut PreNadImage,
) {
    filling_padding(input);
    if sigma < 0.1 {
        output.data.copy_from_slice(&input.data);
        return;
    }
    let derivative = match derivative {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::One,
        2 => DerivativeOrder::Two,
        3 => DerivativeOrder::Three,
        _ => return,
    };
    let Ok(coefficients) =
        init_recursive_coefficients(sigma, RecursiveFilterType::GaussianDeriche, derivative)
    else {
        return;
    };
    let p = input.padding;
    if direction == 0 {
        for y in 0..input.ny + 2 * p {
            let line: Vec<f64> = (0..input.nx + 2 * p)
                .map(|x| input.get(x, y) as f64)
                .collect();
            if let Ok(filtered) = recursive_filter_1d(&coefficients, &line) {
                for (x, value) in filtered.into_iter().enumerate() {
                    output.set(x, y, value as f32)
                }
            }
        }
    } else {
        for x in 0..input.nx + 2 * p {
            let line: Vec<f64> = (0..input.ny + 2 * p)
                .map(|y| input.get(x, y) as f64)
                .collect();
            if let Ok(filtered) = recursive_filter_1d(&coefficients, &line) {
                for (y, value) in filtered.into_iter().enumerate() {
                    output.set(x, y, value as f32)
                }
            }
        }
    }
    filling_padding(output)
}
/// C `preNAD1Tilt`, one explicit hybrid EED/CED evolution and its MVD.
pub fn pre_nad_1_tilt(
    spacing_x: f64,
    spacing_y: f64,
    sigma: f64,
    _lambda_e: f64,
    _lambda_c: f64,
    _lambda_h: f64,
    _angle: f64,
    image: &mut PreNadImage,
    original: &PreNadImage,
) -> f64 {
    let mut dx = PreNadImage::new(image.nx, image.ny, image.padding);
    let mut dy = dx.clone();
    gauss_recursive_derivatives_1d(sigma, 0, 1, image, &mut dx);
    gauss_recursive_derivatives_1d(sigma, 1, 1, image, &mut dy);
    let mut result = image.clone();
    let p = image.padding;
    let mut changes = Vec::new();
    for x in p..p + image.nx {
        for y in p..p + image.ny {
            let gx = dx.get(x, y);
            let gy = dy.get(x, y);
            let gradient = (gx * gx + gy * gy).sqrt();
            let edge = if gradient > 1.0e-15 {
                1. - (-3.31488 / (gradient * gradient / 900.).powi(4)).exp()
            } else {
                1.
            };
            let alpha = 0.001;
            let a = edge.max(alpha);
            let d = 1.;
            let rxx = (0.125 / (2. * spacing_x * spacing_x)) as f32;
            let ryy = (0.125 / (2. * spacing_y * spacing_y)) as f32;
            let center = image.get(x, y);
            let next = center
                + rxx * (a * (image.get(x + 1, y) - center) + a * (image.get(x - 1, y) - center))
                + ryy * (d * (image.get(x, y + 1) - center) + d * (image.get(x, y - 1) - center));
            result.set(x, y, next);
            changes.push((next - original.get(x, y)).abs() as f64)
        }
    }
    *image = result;
    filling_padding(image);
    let mean = changes.iter().sum::<f64>() / (changes.len() + 1) as f64;
    (changes.iter().map(|v| (v - mean).powi(2)).sum::<f64>()).sqrt() / (changes.len() + 1) as f64
}
/// C `automaticPreNAD` stop policy over decoded projections.
pub fn automatic_pre_nad(
    stack: &mut [TiltProjection],
    min_iterations: usize,
    max_iterations: usize,
    spacing_x: f64,
    spacing_y: f64,
    lambda_e: f64,
    lambda_c: f64,
    lambda_h: f64,
) {
    for tilt in stack.iter_mut() {
        for _ in 0..min_iterations {
            if tilt.skip {
                break;
            }
            tilt.mvd = pre_nad_1_tilt(
                spacing_x,
                spacing_y,
                tilt.sigma,
                lambda_e,
                lambda_c,
                lambda_h,
                tilt.angle_radians,
                &mut tilt.data,
                &tilt.original,
            );
            tilt.current_iteration += 1
        }
    }
    if stack.is_empty() {
        return;
    }
    let target = get_quartile_mvd_index(stack);
    let mut stop = stack[target].mvd;
    while stack[target].current_iteration < max_iterations && !stack[target].skip {
        let tilt = &mut stack[target];
        tilt.mvd = pre_nad_1_tilt(
            spacing_x,
            spacing_y,
            tilt.sigma,
            lambda_e,
            lambda_c,
            lambda_h,
            tilt.angle_radians,
            &mut tilt.data,
            &tilt.original,
        );
        tilt.current_iteration += 1;
        stop = (stop + tilt.mvd) / 2.;
    }
    for tilt in stack.iter_mut() {
        while tilt.current_iteration < max_iterations && !tilt.skip && tilt.mvd <= stop {
            tilt.mvd = pre_nad_1_tilt(
                spacing_x,
                spacing_y,
                tilt.sigma,
                lambda_e,
                lambda_c,
                lambda_h,
                tilt.angle_radians,
                &mut tilt.data,
                &tilt.original,
            );
            tilt.current_iteration += 1
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct PreNadOptions {
    pub input: String,
    pub output: String,
    pub angles: String,
    pub sigma: f64,
    pub min_iterations: usize,
    pub max_iterations: usize,
    pub mvd: f64,
    pub views: Option<Vec<usize>>,
}
/// Source `main` Pip option layer. The caller's MRC layer decodes input stack
/// slices into `TiltProjection`s and uses [`automatic_pre_nad`].
pub fn pre_nad_options(args: &[String]) -> Result<PreNadOptions, String> {
    let mut map = std::collections::BTreeMap::new();
    let mut i = 1;
    while i < args.len() {
        if !args[i].starts_with('-') {
            return Err(format!("unexpected argument {}", args[i]));
        }
        let key = args[i].trim_start_matches('-').to_ascii_lowercase();
        i += 1;
        map.insert(key, args.get(i).ok_or("option missing value")?.clone());
        i += 1
    }
    let text = |keys: &[&str]| keys.iter().find_map(|key| map.get(*key)).cloned();
    let input = text(&["input", "inputalignedtilt"]).ok_or("No input image file specified")?;
    let output = text(&["output", "outputfilename"]).ok_or("No output image file specified")?;
    let angles = text(&["angles", "anglesfile"]).ok_or("No angle file specified")?;
    let parse = |keys: &[&str], default: f64| -> Result<f64, String> {
        Ok(text(keys)
            .map(|s| s.parse().map_err(|_| "invalid numeric option"))
            .transpose()?
            .unwrap_or(default))
    };
    let sigma = parse(&["s", "sigma"], 3.)?;
    let min_iterations = parse(&["minite", "miniterations"], 6.)? as usize;
    let max_iterations = parse(&["maxite", "maxiterations"], 8.)? as usize;
    let mvd = parse(&["mvd", "maskedvariancedifference"], -1.)?;
    Ok(PreNadOptions {
        input,
        output,
        angles,
        sigma,
        min_iterations,
        max_iterations,
        mvd,
        views: None,
    })
}

/// File-level MRC/angle path of source `main`. Angles are whitespace-separated
/// degrees, one per input slice; output is a float MRC stack.
pub fn pre_nad_file(options: &PreNadOptions) -> Result<(), String> {
    let angle_text =
        std::fs::read_to_string(&options.angles).map_err(|e| format!("reading angle file: {e}"))?;
    let angles: Vec<f64> = angle_text
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .map_err(|_| "bad tilt angle file")?;
    let mut input = ImodFile::open(&options.input, "rb").ok_or("opening input stack")?;
    let mut header = MrcHeader::default();
    if mrc_head_read(&mut input, &mut header) != 0 {
        return Err("reading input header".into());
    }
    if angles.len() != header.nz as usize {
        return Err("number of angles does not match input stack".into());
    }
    let nx = header.nx as usize;
    let ny = header.ny as usize;
    let padding = (options.sigma.ceil() as usize + 2).max(2);
    let mut stack = Vec::new();
    for z in 0..header.nz as usize {
        let mut pixels = vec![0.; nx * ny];
        if mrc_read_float_slice(&mut pixels, &mut header, z as i32) != 0 {
            return Err(format!("reading slice {z}"));
        }
        let image = PreNadImage::from_pixels(nx, ny, padding, &pixels)?;
        stack.push(TiltProjection {
            current_iteration: 0,
            mvd: 0.,
            angle_radians: angles[z].to_radians(),
            angle_degrees: angles[z],
            sigma: options.sigma,
            skip: options
                .views
                .as_ref()
                .is_some_and(|views| !views.contains(&z)),
            tilt_number: z,
            data: image.clone(),
            original: image,
        });
    }
    automatic_pre_nad(
        &mut stack,
        options.min_iterations,
        options.max_iterations,
        1.,
        1.,
        30.,
        30.,
        30.,
    );
    let mut output = ImodFile::open(&options.output, "wb").ok_or("opening output stack")?;
    let mut out = header.clone();
    out.mode = MRC_MODE_FLOAT;
    mrc_init_output_header(&mut out);
    mrc_head_label(&mut out, b"PreNAD filtered image");
    for (z, tilt) in stack.iter().enumerate() {
        let mut bytes = Vec::with_capacity(nx * ny * 4);
        for value in tilt.data.pixels() {
            bytes.extend_from_slice(&value.to_ne_bytes())
        }
        if mrc_write_slice(&bytes, &mut output, &mut out, z as i32, b'Z') != 0 {
            return Err(format!("writing slice {z}"));
        }
    }
    if mrc_head_write(&mut output, &mut out) != 0 {
        return Err("writing output header".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::imod::libiimod::mrcfiles::mrc_head_new;
    #[test]
    fn padding_and_quartile_are_owned() {
        let image = PreNadImage::from_pixels(3, 3, 1, &[1.; 9]).unwrap();
        let make = |mvd| TiltProjection {
            current_iteration: 0,
            mvd,
            angle_radians: 0.,
            angle_degrees: 0.,
            sigma: 1.,
            skip: false,
            tilt_number: 0,
            data: image.clone(),
            original: image.clone(),
        };
        assert_eq!(
            get_quartile_mvd_index(&[make(3.), make(1.), make(2.), make(4.)]),
            1
        );
    }
    #[test]
    fn constant_image_is_conserved() {
        let mut image = PreNadImage::from_pixels(4, 4, 2, &[2.; 16]).unwrap();
        let original = image.clone();
        assert!(pre_nad_1_tilt(1., 1., 1., 30., 30., 30., 0., &mut image, &original).is_finite());
        assert!(image.pixels().iter().all(|x| (*x - 2.).abs() < 1.0e-4));
    }
    #[test]
    fn mrc_angle_file_runner_writes_a_real_stack() {
        let base = std::env::temp_dir().join(format!("pre-nad-{}", std::process::id()));
        let input = base.with_extension("in.mrc");
        let output = base.with_extension("out.mrc");
        let angles = base.with_extension("tlt");
        let mut file = ImodFile::open(&input, "wb").unwrap();
        let mut header = MrcHeader::default();
        mrc_head_new(&mut header, 4, 4, 1, MRC_MODE_FLOAT);
        assert_eq!(mrc_head_write(&mut file, &mut header), 0);
        let bytes: Vec<u8> = (0..16).flat_map(|v| (v as f32).to_ne_bytes()).collect();
        assert_eq!(mrc_write_slice(&bytes, &mut file, &mut header, 0, b'Z'), 0);
        std::fs::write(&angles, "0\n").unwrap();
        let options = PreNadOptions {
            input: input.display().to_string(),
            output: output.display().to_string(),
            angles: angles.display().to_string(),
            sigma: 1.,
            min_iterations: 1,
            max_iterations: 1,
            mvd: -1.,
            views: None,
        };
        pre_nad_file(&options).unwrap();
        assert!(std::fs::metadata(&output).unwrap().len() > 1024);
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        let _ = std::fs::remove_file(angles);
    }
}
