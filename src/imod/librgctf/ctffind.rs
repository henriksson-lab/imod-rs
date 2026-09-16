//! Bottom-up native translation of `IMOD/librgctf/ctffind.cpp`.

use super::empirical_distribution::EmpiricalDistribution;
use super::functions::rank_sort;
use super::{brute_force_search::BruteForceSearch, conjugate_gradient::ConjugateGradient};
use super::{ctf::Ctf, curve::Curve, image::Image};

/// Parameters declared by `IMOD/include/ctffind.h`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CtffindParams {
    pub pixel_size_of_input_image: f32,
    pub acceleration_voltage: f32,
    pub spherical_aberration: f32,
    pub amplitude_contrast: f32,
    pub box_size: i32,
    pub minimum_resolution: f32,
    pub maximum_resolution: f32,
    pub minimum_defocus: f32,
    pub maximum_defocus: f32,
    pub defocus_search_step: f32,
    pub slower_search: bool,
    pub astigmatism_tolerance: f32,
    pub find_additional_phase_shift: bool,
    pub minimum_additional_phase_shift: f32,
    pub maximum_additional_phase_shift: f32,
    pub additional_phase_shift_search_step: f32,
    pub astigmatism_is_known: bool,
    pub known_astigmatism: f32,
    pub known_astigmatism_angle: f32,
    pub compute_extra_stats: bool,
    pub noisy_input_image: bool,
}

/// Owned result of the exported source `ctffind` routine.
///
/// This replaces its caller-allocated result array and optional malloc output
/// arrays with one value whose vectors are always released by Rust.
#[derive(Clone, Debug, PartialEq)]
pub struct CtffindResult {
    pub defocus_1: f32,
    pub defocus_2: f32,
    pub astigmatism_angle_degrees: f32,
    pub additional_phase_shift: f32,
    pub score: f32,
    pub fit_resolution: f32,
    pub aliasing_resolution: f32,
    pub rotational_average: Vec<f32>,
    pub normalized_average: Option<Vec<f32>>,
    pub fit_curve: Option<Vec<f32>>,
    pub last_bin_frequency: f32,
}

/// Safe, owned translation of the exported `ctffind` entry point.
pub fn ctffind(
    params: &CtffindParams,
    spectrum_array: &[f32],
    input_stride: usize,
) -> Result<CtffindResult, String> {
    let box_size = usize::try_from(params.box_size).map_err(|_| "negative box size")?;
    if box_size < 2 || input_stride < box_size || spectrum_array.len() < input_stride * box_size {
        return Err("input spectrum dimensions are invalid".into());
    }
    if params.minimum_resolution < params.maximum_resolution {
        return Err("minimum resolution must be at least maximum resolution".into());
    }
    if params.minimum_defocus > params.maximum_defocus || params.pixel_size_of_input_image <= 0. {
        return Err("invalid defocus range or pixel size".into());
    }
    let pixel_size = params.pixel_size_of_input_image;
    let fixed_phase =
        (params.maximum_additional_phase_shift - params.minimum_additional_phase_shift).abs()
            < 0.01;
    let mut spectrum = Image::default();
    spectrum.allocate(box_size as i32, box_size as i32, 1, true)?;
    for y in 0..box_size as i32 {
        for x in 0..box_size as i32 {
            spectrum.set_real_pixel_from_physical_coord(
                x,
                y,
                0,
                spectrum_array[y as usize * input_stride + x as usize],
            );
        }
    }
    spectrum.zero_central_pixel();
    let minimum_radius = box_size as f32 * pixel_size / params.minimum_resolution;
    let maximum_radius = box_size as f32;
    let (average, sigma) = spectrum
        .compute_average_and_sigma_of_values_in_spectrum(minimum_radius, maximum_radius, 12)
        .ok_or("no values available for spectrum normalization")?;
    if sigma == 0. {
        return Err("constant spectrum cannot be fitted".into());
    }
    spectrum.divide_by_constant(sigma);
    spectrum.set_maximum_value_on_central_cross(average / sigma + 10.);
    let mut background = Image::default();
    background.allocate(box_size as i32, box_size as i32, 1, true)?;
    let mut convolution_size = (minimum_radius * 2_f32.sqrt()) as i32;
    if convolution_size % 2 == 0 {
        convolution_size += 1;
    }
    background.set_to_constant(0.);
    spectrum.spectrum_box_convolution(&mut background, convolution_size, minimum_radius);
    spectrum.subtract_image(&background);
    spectrum.set_maximum_value(spectrum.return_maximum_value(3., 3.));
    let mut masked = Image::default();
    masked.copy_from(&spectrum);
    if !params.noisy_input_image {
        masked.cosine_mask(
            box_size as f32 * pixel_size / params.maximum_resolution.max(8.),
            box_size as f32 * pixel_size / params.maximum_resolution.max(4.),
            true,
            None,
        );
    }
    let mut ctf = Ctf::default();
    ctf.init_with_fitting_parameters(
        params.acceleration_voltage,
        params.spherical_aberration,
        params.amplitude_contrast,
        params.minimum_defocus,
        params.minimum_defocus,
        0.,
        1. / params.minimum_resolution,
        1. / params.maximum_resolution.max(5.),
        params.astigmatism_tolerance,
        pixel_size,
        params.minimum_additional_phase_shift,
    );
    ctf.set_defocus(
        params.minimum_defocus / pixel_size,
        params.minimum_defocus / pixel_size,
        0.,
    );
    let mut comparison = ImageCtfComparison::new(
        1,
        ctf.clone(),
        pixel_size,
        params.find_additional_phase_shift && !fixed_phase,
        params.astigmatism_is_known,
        params.known_astigmatism / pixel_size,
        params.known_astigmatism_angle.to_radians(),
        false,
    );
    comparison.set_image(0, &masked);
    comparison.setup_quick_correlation();
    let estimated_angle = if params.astigmatism_is_known {
        params.known_astigmatism_angle
    } else {
        let mut mirrored = Image::default();
        mirrored.copy_from(&spectrum);
        mirrored.apply_mirror_along_y();
        0.5 * find_rotational_alignment_between_two_stacks_of_images(
            &[spectrum.clone()],
            &[mirrored],
            90.,
            5.,
            pixel_size / params.minimum_resolution,
            pixel_size / params.maximum_resolution.max(5.),
        )
    };
    let phase_dimensions = usize::from(params.find_additional_phase_shift && !fixed_phase);
    if !params.slower_search {
        let bins = masked.return_maximum_diagonal_radius().ceil() as usize;
        let mut curve = Curve::new();
        curve.setup_x_axis(0., bins as f32 * masked.fourier_voxel_size_x, bins);
        let mut counts = curve.clone();
        masked.compute_1d_rotational_average(&mut curve, &mut counts, true);
        let comparison_1d = CurveCtfComparison {
            curve: curve.data_y.clone(),
            reciprocal_pixel_size: masked.fourier_voxel_size_x,
            ctf: ctf.clone(),
            find_phase_shift: phase_dimensions == 1,
        };
        let midpoint = vec![
            (params.minimum_defocus + params.maximum_defocus) * 0.5 / pixel_size,
            (params.minimum_additional_phase_shift + params.maximum_additional_phase_shift) * 0.5,
        ];
        let half = vec![
            (params.maximum_defocus - params.minimum_defocus) * 0.5 / pixel_size,
            (params.maximum_additional_phase_shift - params.minimum_additional_phase_shift) * 0.5,
        ];
        let step = vec![
            params.defocus_search_step / pixel_size,
            params.additional_phase_shift_search_step,
        ];
        let dimensions = 1 + phase_dimensions;
        let owned = comparison_1d.clone();
        let mut search = BruteForceSearch::new();
        search.init(
            move |values| ctffind_curve_objective_function(&owned, values),
            &midpoint[..dimensions],
            &half[..dimensions],
            &step[..dimensions],
            false,
            false,
        );
        search.run();
        let mut start = (0..dimensions)
            .map(|i| search.get_best_value(i))
            .collect::<Vec<_>>();
        let owned = comparison_1d.clone();
        let mut minimizer = ConjugateGradient::new();
        minimizer
            .init(
                move |values| ctffind_curve_objective_function(&owned, values),
                &start,
                &[100., 0.05][..dimensions],
            )
            .map_err(|error| format!("1D minimizer: {error:?}"))?;
        minimizer
            .run()
            .map_err(|error| format!("1D minimizer: {error:?}"))?;
        start.clone_from_slice(minimizer.best_values());
        ctf.set_defocus(start[0], start[0], estimated_angle.to_radians());
        if params.find_additional_phase_shift {
            ctf.set_additional_phase_shift(if fixed_phase {
                params.minimum_additional_phase_shift
            } else {
                start[1]
            });
        }
    }
    let dimensions = if params.astigmatism_is_known {
        1 + phase_dimensions
    } else {
        3 + phase_dimensions
    };
    let mut start = if params.astigmatism_is_known {
        vec![ctf.defocus_1()]
    } else {
        vec![ctf.defocus_1(), ctf.defocus_2(), ctf.astigmatism_azimuth()]
    };
    if phase_dimensions == 1 {
        start.push(ctf.additional_phase_shift());
    }
    let accuracy = if params.astigmatism_is_known {
        vec![100., 0.05]
    } else {
        vec![100., 100., 0.025, 0.05]
    };
    comparison.set_ctf(ctf.clone());
    let owned = comparison.clone();
    let mut minimizer = ConjugateGradient::new();
    minimizer
        .init(
            move |values| ctffind_objective_function(&owned, values),
            &start,
            &accuracy[..dimensions],
        )
        .map_err(|error| format!("2D minimizer: {error:?}"))?;
    minimizer
        .run()
        .map_err(|error| format!("2D minimizer: {error:?}"))?;
    let values = minimizer.best_values();
    if params.astigmatism_is_known {
        ctf.set_defocus(
            values[0],
            values[0] - params.known_astigmatism / pixel_size,
            params.known_astigmatism_angle.to_radians(),
        );
    } else {
        ctf.set_defocus(values[0], values[1], values[2]);
    }
    if params.find_additional_phase_shift {
        ctf.set_additional_phase_shift(if fixed_phase {
            params.minimum_additional_phase_shift
        } else {
            values[dimensions - 1]
        });
    }
    ctf.enforce_convention();
    let bins = spectrum.return_maximum_diagonal_radius().ceil() as usize;
    let mut regular = Curve::new();
    regular.setup_x_axis(0., bins as f32 * spectrum.fourier_voxel_size_x, bins);
    let mut counts = regular.clone();
    spectrum.compute_1d_rotational_average(&mut regular, &mut counts, true);
    let mut normalized_average = None;
    let mut fit_curve = None;
    let mut fit_resolution = 0.;
    let mut aliasing_resolution = 0.;
    if params.compute_extra_stats {
        let mut extrema = Image::default();
        let mut values_image = Image::default();
        extrema.allocate(box_size as i32, box_size as i32, 1, true)?;
        values_image.allocate(box_size as i32, box_size as i32, 1, true)?;
        compute_images_with_number_of_extrema_and_ctf_values(&ctf, &mut extrema, &mut values_image);
        let rotational = compute_rotational_average_of_power_spectrum(
            &spectrum,
            &ctf,
            &extrema,
            &values_image,
            bins,
        )
        .ok_or("unable to bin rotational average")?;
        let first = rotational
            .spatial_frequency
            .iter()
            .position(|&f| f as f32 >= ctf.lowest_frequency_for_fitting())
            .unwrap_or(0);
        let (frc, _) = compute_frc_between_1d_spectrum_and_fit(
            &rotational.average_rank,
            &rotational.average_fit,
            &rotational.extrema_profile,
            first,
        );
        let last = frc.iter().rposition(|&value| value >= 0.1).unwrap_or(0);
        if last > 0 {
            fit_resolution = pixel_size / rotational.spatial_frequency[last] as f32;
        }
        let alias = (1..bins)
            .find(|&i| {
                rotational.extrema_profile[i] - rotational.extrema_profile[i - 1] >= 0.9 && i > 3
            })
            .unwrap_or(0);
        if alias > 0 {
            aliasing_resolution = pixel_size / rotational.spatial_frequency[alias] as f32;
        }
        regular.data_y = rotational.average.iter().map(|&v| v as f32).collect();
        normalized_average = Some(rotational.average_rank.iter().map(|&v| v as f32).collect());
        fit_curve = Some(rotational.average_fit.iter().map(|&v| v as f32).collect());
    }
    Ok(CtffindResult {
        defocus_1: ctf.defocus_1() * pixel_size,
        defocus_2: ctf.defocus_2() * pixel_size,
        astigmatism_angle_degrees: ctf.astigmatism_azimuth().to_degrees(),
        additional_phase_shift: ctf.additional_phase_shift(),
        score: -minimizer.best_score(),
        fit_resolution,
        aliasing_resolution,
        rotational_average: regular.data_y,
        normalized_average,
        fit_curve,
        last_bin_frequency: (bins.saturating_sub(1)) as f32 / box_size as f32,
    })
}

/// Owned form of source `ImageCTFComparison`.
#[derive(Clone, Debug)]
pub struct ImageCtfComparison {
    pub images: Vec<Image>,
    ctf: Ctf,
    pub pixel_size: f32,
    find_phase_shift: bool,
    astigmatism_is_known: bool,
    known_astigmatism: f32,
    known_astigmatism_angle: f32,
    pub fit_defocus_sweep: bool,
    addresses: Vec<usize>,
    spatial_frequency_squared: Vec<f32>,
    azimuths: Vec<f32>,
    norm_image: f64,
    image_mean: f64,
}

impl ImageCtfComparison {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        number_of_images: usize,
        ctf: Ctf,
        pixel_size: f32,
        find_phase_shift: bool,
        astigmatism_is_known: bool,
        known_astigmatism: f32,
        known_astigmatism_angle: f32,
        fit_defocus_sweep: bool,
    ) -> Self {
        Self {
            images: vec![Image::default(); number_of_images],
            ctf,
            pixel_size,
            find_phase_shift,
            astigmatism_is_known,
            known_astigmatism,
            known_astigmatism_angle,
            fit_defocus_sweep,
            addresses: Vec::new(),
            spatial_frequency_squared: Vec::new(),
            azimuths: Vec::new(),
            norm_image: 0.,
            image_mean: 0.,
        }
    }
    pub fn set_image(&mut self, image_number: usize, image: &Image) {
        self.images[image_number].copy_from(image);
    }
    pub fn set_ctf(&mut self, ctf: Ctf) {
        self.ctf = ctf;
    }
    pub fn ctf(&self) -> &Ctf {
        &self.ctf
    }
    pub fn astigmatism_is_known(&self) -> bool {
        self.astigmatism_is_known
    }
    pub fn known_astigmatism(&self) -> f32 {
        self.known_astigmatism
    }
    pub fn known_astigmatism_angle(&self) -> f32 {
        self.known_astigmatism_angle
    }
    pub fn find_phase_shift(&self) -> bool {
        self.find_phase_shift
    }
    pub fn setup_quick_correlation(&mut self) {
        let (addresses, frequencies, azimuths, norm, mean) =
            self.images[0].setup_quick_correlation_with_ctf(&self.ctf);
        self.addresses = addresses;
        self.spatial_frequency_squared = frequencies;
        self.azimuths = azimuths;
        self.norm_image = norm;
        self.image_mean = mean;
    }
}

/// Source `CurveCTFComparison`, represented with an owned curve sample vector.
#[derive(Clone, Debug)]
pub struct CurveCtfComparison {
    pub curve: Vec<f32>,
    pub reciprocal_pixel_size: f32,
    pub ctf: Ctf,
    pub find_phase_shift: bool,
}

/// `CtffindObjectiveFunction`.
pub fn ctffind_objective_function(comparison: &ImageCtfComparison, values: &[f32]) -> f32 {
    let mut ctf = comparison.ctf.clone();
    if comparison.astigmatism_is_known {
        assert!(comparison.known_astigmatism >= 0.);
        ctf.set_defocus(
            values[0],
            values[0] - comparison.known_astigmatism,
            comparison.known_astigmatism_angle,
        );
    } else {
        ctf.set_defocus(values[0], values[1], values[2]);
    }
    if comparison.find_phase_shift {
        ctf.set_additional_phase_shift(if comparison.astigmatism_is_known {
            values[1]
        } else {
            values[3]
        });
    }
    if comparison.addresses.is_empty() {
        -comparison.images[0].get_correlation_with_ctf(&ctf)
    } else {
        -comparison.images[0].quick_correlation_with_ctf(
            &ctf,
            &comparison.addresses,
            &comparison.spatial_frequency_squared,
            &comparison.azimuths,
            comparison.norm_image,
            comparison.image_mean,
        )
    }
}

/// `CtffindCurveObjectiveFunction`.
pub fn ctffind_curve_objective_function(comparison: &CurveCtfComparison, values: &[f32]) -> f32 {
    let mut ctf = comparison.ctf.clone();
    ctf.set_defocus(values[0], values[0], 0.);
    if comparison.find_phase_shift {
        ctf.set_additional_phase_shift(values[1]);
    }
    let lowest = ctf.lowest_frequency_for_fitting().powi(2);
    let highest = ctf.highest_frequency_for_fitting().powi(2);
    let mut cross = 0_f64;
    let mut curve_norm = 0_f64;
    let mut ctf_norm = 0_f64;
    for (bin, &sample) in comparison.curve.iter().enumerate() {
        let frequency_squared = (bin as f32 * comparison.reciprocal_pixel_size).powi(2);
        if frequency_squared > lowest && frequency_squared < highest {
            let ctf_value = ctf.evaluate(frequency_squared, 0.).abs();
            cross += (sample * ctf_value) as f64;
            curve_norm += sample.powi(2) as f64;
            ctf_norm += ctf_value.powi(2) as f64;
        }
    }
    -(cross / (curve_norm * ctf_norm).sqrt()) as f32
}

/// `Renormalize1DSpectrumForFRC`.
pub fn renormalize_1d_spectrum_for_frc(average: &mut [f64], fit: &[f64], extrema: &[f32]) {
    assert_eq!(average.len(), fit.len());
    assert_eq!(average.len(), extrema.len());
    let mut previous = 0usize;
    for bin in 1..average.len() {
        if extrema[bin] - extrema[bin - 1] >= 0.9 {
            let current = bin - 1;
            if previous > 0 {
                if extrema[bin] < 7. {
                    let mut zero = (current - previous) / 2 + previous;
                    for i in previous..current {
                        if fit[i] < fit[i - 1] && fit[i] < fit[i + 1] {
                            zero = i;
                        }
                    }
                    let values = average[previous..=zero]
                        .iter()
                        .map(|&value| value as f32)
                        .collect::<Vec<_>>();
                    let ranks = rank_sort(&values);
                    for i in previous..=zero {
                        average[i] = ((ranks[i - previous] as f64 / (values.len() - 1) as f64)
                            * std::f64::consts::FRAC_PI_2)
                            .sin();
                    }
                    let values = average[zero + 1..current]
                        .iter()
                        .map(|&value| value as f32)
                        .collect::<Vec<_>>();
                    if !values.is_empty() {
                        let ranks = rank_sort(&values);
                        for i in zero + 1..current {
                            average[i] = (((ranks[i - zero - 1] + 1) as f64
                                / (values.len() + 1) as f64)
                                * std::f64::consts::FRAC_PI_2)
                                .sin();
                        }
                    }
                } else {
                    let (minimum, maximum) = average[previous..current]
                        .iter()
                        .fold((1_f64, 0_f64), |(min, max), &value| {
                            (min.min(value), max.max(value))
                        });
                    for value in &mut average[previous..current] {
                        *value -= minimum;
                        if maximum - minimum > 0.0001 {
                            *value /= maximum - minimum;
                        }
                    }
                }
            }
            previous = current;
        }
    }
}

/// `ComputeFRCBetween1DSpectrumAndFit`.
pub fn compute_frc_between_1d_spectrum_and_fit(
    average: &[f64],
    fit: &[f64],
    extrema: &[f32],
    first_fit_bin: usize,
) -> (Vec<f64>, Vec<f64>) {
    let bins = average.len();
    assert_eq!(bins, fit.len());
    assert_eq!(bins, extrema.len());
    assert!(bins >= 3 && first_fit_bin < bins);
    let minimum = bins / 40;
    let mut half_width = vec![minimum; bins];
    let mut previous = 0;
    for bin in 1..bins {
        if extrema[bin] != extrema[bin - 1] {
            let width = (((1. + 0.1 * extrema[bin]) * (bin - previous + 1) as f32) as usize)
                .max(minimum)
                .min(bins / 2 - 1);
            half_width[previous..bin].fill(width);
            previous = bin;
        }
    }
    half_width[0] = half_width[1];
    if previous > 0 {
        let width = half_width[previous - 1];
        half_width[previous..].fill(width);
    }
    let mut frc = vec![0.; bins];
    let mut sigma = vec![0.; bins];
    for bin in 0..bins {
        if bin < first_fit_bin {
            frc[bin] = 1.;
            continue;
        }
        let width = half_width[bin];
        let mut first = bin.saturating_sub(width).max(first_fit_bin);
        let mut last = (bin + width).min(bins - 1);
        if first == first_fit_bin {
            last = (first + 2 * width + 1).min(bins - 1);
        }
        if last == bins - 1 {
            first = last.saturating_sub(2 * width + 1).max(first_fit_bin);
        }
        let count = (last - first + 1) as f64;
        let mean_a = average[first..=last].iter().sum::<f64>() / count;
        let mean_f = fit[first..=last].iter().sum::<f64>() / count;
        let (cross, norm_a, norm_f) = (first..=last).fold((0., 0., 0.), |(cross, na, nf), i| {
            let a = average[i] - mean_a;
            let f = fit[i] - mean_f;
            (cross + a * f, na + a * a, nf + f * f)
        });
        frc[bin] = if norm_a > 0. && norm_f > 0. {
            cross / ((norm_a / count).sqrt() * (norm_f / count).sqrt()) / count
        } else {
            0.
        };
        sigma[bin] = 2. / count.sqrt();
    }
    (frc, sigma)
}

/// `OverlayCTF`, including the source's lower-left theoretical overlay.
pub fn overlay_ctf(spectrum: &mut Image, ctf: &Ctf) {
    assert!(spectrum.is_in_real_space && spectrum.logical_z_dimension == 1);
    let lowest = ctf.lowest_frequency_for_fitting().powi(2);
    let highest = ctf.highest_frequency_for_fitting().powi(2);
    for y in 0..spectrum.logical_y_dimension {
        let fy =
            (y - spectrum.physical_address_of_box_center_y) as f32 * spectrum.fourier_voxel_size_y;
        for x in 0..spectrum.logical_x_dimension {
            let fx = (x - spectrum.physical_address_of_box_center_x) as f32
                * spectrum.fourier_voxel_size_x;
            let frequency_squared = fx.powi(2) + fy.powi(2);
            if frequency_squared > lowest
                && frequency_squared <= highest
                && y < spectrum.physical_address_of_box_center_y
                && x < spectrum.physical_address_of_box_center_x
            {
                spectrum.set_real_pixel_from_physical_coord(
                    x,
                    y,
                    0,
                    ctf.evaluate(frequency_squared, fy.atan2(fx)).abs(),
                );
            }
            if frequency_squared <= lowest {
                spectrum.set_real_pixel_from_physical_coord(x, y, 0, 0.);
            }
        }
    }
}

/// Live one-dimensional branch of `RescaleSpectrumAndRotationalAverage`.
/// The source's spectrum-pixel branch is explicitly disabled by IMOD.
pub fn rescale_rotational_average(
    average: &mut [f64],
    fit: &[f64],
    spatial_frequency: &[f64],
    last_bin_without_aliasing: usize,
) -> bool {
    if average.len() <= 1 || average.len() != fit.len() || average.len() != spatial_frequency.len()
    {
        return false;
    }
    let bins = average.len();
    let mut background = vec![0.; bins];
    let mut peak = vec![0.; bins];
    let mut minima = Curve::new();
    let mut maxima = Curve::new();
    let (mut previous_maximum, mut previous_minimum, mut maximum_number, mut normalization) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut at_maximum, mut at_minimum) = (false, true);
    for bin in 1..bins - 1 {
        let (was_maximum, was_minimum) = (at_maximum, at_minimum);
        at_minimum = fit[bin] <= fit[bin - 1] && fit[bin] <= fit[bin + 1];
        at_maximum = fit[bin] >= fit[bin - 1] && fit[bin] >= fit[bin + 1];
        if at_maximum && at_minimum {
            at_maximum = was_maximum;
            at_minimum = was_minimum;
        }
        if at_minimum {
            for i in previous_minimum + 1..=bin {
                background[i] = average[previous_minimum] * (bin - i) as f64
                    / (bin - previous_minimum) as f64
                    + average[bin] * (i - previous_minimum) as f64
                        / (bin - previous_minimum) as f64;
            }
            previous_minimum = bin;
            minima.add_point(spatial_frequency[bin] as f32, average[bin] as f32);
        }
        if at_maximum {
            if !was_maximum && fit[bin] > 0.7 {
                maximum_number += 1;
            }
            for i in previous_maximum + 1..=bin {
                peak[i] = average[previous_maximum] * (bin - i) as f64
                    / (bin - previous_maximum) as f64
                    + average[bin] * (i - previous_maximum) as f64
                        / (bin - previous_maximum) as f64;
                if maximum_number == 2 {
                    normalization = bin;
                }
            }
            previous_maximum = bin;
            maxima.add_point(spatial_frequency[bin] as f32, average[bin] as f32);
        }
        if at_maximum && at_minimum {
            return false;
        }
    }
    if minima.number_of_points > 7 {
        minima.fit_savitzky_golay_to_data(7, 2);
    }
    if maxima.number_of_points > 7 {
        maxima.fit_savitzky_golay_to_data(7, 2);
    }
    for bin in 0..bins {
        if minima.number_of_points > 7 {
            background[bin] = minima
                .return_savitzky_golay_interpolation_from_x(spatial_frequency[bin] as f32)
                as f64;
        }
        if maxima.number_of_points > 7 {
            peak[bin] = maxima
                .return_savitzky_golay_interpolation_from_x(spatial_frequency[bin] as f32)
                as f64;
        }
    }
    let scale = peak[normalization] - background[normalization];
    if scale > 0. {
        for bin in 0..bins {
            average[bin] = (average[bin] - background[bin]) / scale * 0.95;
            let local = peak[bin] - background[bin];
            if local < 0.1 && local.abs() > 0.000001 && bin <= last_bin_without_aliasing {
                average[bin] = average[bin] / local * scale * 0.1;
            }
        }
    }
    true
}

/// `ReturnSpectrumBinNumber` using an owned physical-image address.
pub fn return_spectrum_bin_number(
    extrema_profile: &[f32],
    extrema_value: f32,
    ctf_value: f32,
    ctf_profile: &[f32],
) -> Option<usize> {
    if extrema_profile.is_empty() || extrema_profile.len() != ctf_profile.len() {
        return None;
    }
    let mut ctf_difference = f32::MAX;
    let mut chosen = None;
    for bin in 0..extrema_profile.len() {
        let difference = (extrema_value - extrema_profile[bin]).abs();
        let previous = if bin > 0 {
            (extrema_value - extrema_profile[bin - 1]).abs()
        } else {
            f32::MAX
        };
        let next = if bin + 1 < extrema_profile.len() {
            (extrema_value - extrema_profile[bin + 1]).abs()
        } else {
            f32::MAX
        };
        if extrema_value > extrema_profile[extrema_profile.len() - 1] {
            chosen = Some(extrema_profile.len() - 1);
        } else if difference <= 0.01
            || (difference < previous
                && difference <= next
                && extrema_profile[bin.saturating_sub(1)]
                    != extrema_profile[(bin + 1).min(extrema_profile.len() - 1)])
        {
            let candidate = (ctf_value - ctf_profile[bin]).abs();
            if candidate < ctf_difference {
                ctf_difference = candidate;
                chosen = Some(bin);
            }
        }
    }
    chosen
}

/// `ComputeImagesWithNumberOfExtremaAndCTFValues`.
pub fn compute_images_with_number_of_extrema_and_ctf_values(
    ctf: &Ctf,
    extrema: &mut Image,
    ctf_values: &mut Image,
) {
    assert!(extrema.has_same_dimensions_as(ctf_values) && extrema.logical_z_dimension == 1);
    for y in 0..extrema.logical_y_dimension {
        let fy =
            (y - extrema.physical_address_of_box_center_y) as f32 * extrema.fourier_voxel_size_y;
        for x in 0..extrema.logical_x_dimension {
            let fx = (x - extrema.physical_address_of_box_center_x) as f32
                * extrema.fourier_voxel_size_x;
            let frequency_squared = fx.powi(2) + fy.powi(2);
            let azimuth = if frequency_squared > 0. {
                fy.atan2(fx)
            } else {
                0.
            };
            ctf_values.set_real_pixel_from_physical_coord(
                x,
                y,
                0,
                ctf.evaluate(frequency_squared, azimuth),
            );
            extrema.set_real_pixel_from_physical_coord(
                x,
                y,
                0,
                ctf.number_of_extrema_before_squared_spatial_frequency(frequency_squared, azimuth)
                    as f32,
            );
        }
    }
    extrema.is_in_real_space = true;
    ctf_values.is_in_real_space = true;
}

/// Output arrays produced by `ComputeRotationalAverageOfPowerSpectrum`.
#[derive(Clone, Debug, PartialEq)]
pub struct RotationalAverage {
    pub spatial_frequency: Vec<f64>,
    pub average: Vec<f64>,
    pub average_fit: Vec<f64>,
    pub average_rank: Vec<f64>,
    pub extrema_profile: Vec<f32>,
    pub ctf_profile: Vec<f32>,
}

/// `ComputeRotationalAverageOfPowerSpectrum`.
pub fn compute_rotational_average_of_power_spectrum(
    spectrum: &Image,
    ctf: &Ctf,
    extrema: &Image,
    ctf_values: &Image,
    number_of_bins: usize,
) -> Option<RotationalAverage> {
    assert!(spectrum.is_in_real_space && extrema.is_in_real_space && ctf_values.is_in_real_space);
    assert!(spectrum.has_same_dimensions_as(extrema));
    assert!(spectrum.has_same_dimensions_as(ctf_values));
    if number_of_bins == 0 {
        return None;
    }
    let mut result = RotationalAverage {
        spatial_frequency: vec![0.; number_of_bins],
        average: vec![0.; number_of_bins],
        average_fit: vec![0.; number_of_bins],
        average_rank: vec![0.; number_of_bins],
        extrema_profile: vec![0.; number_of_bins],
        ctf_profile: vec![0.; number_of_bins],
    };
    if !spectrum.is_constant() {
        let minimum_angular_distance = 10. / 180. * std::f32::consts::PI;
        let mut azimuth = ctf.astigmatism_azimuth() + std::f32::consts::FRAC_PI_4;
        let distance_from_axis = azimuth % std::f32::consts::FRAC_PI_2;
        if distance_from_axis.abs() < minimum_angular_distance {
            azimuth = if distance_from_axis > 0. {
                minimum_angular_distance
            } else {
                -minimum_angular_distance
            };
        }
        if distance_from_axis.abs() > std::f32::consts::FRAC_PI_2 - minimum_angular_distance {
            azimuth = if distance_from_axis > 0. {
                std::f32::consts::FRAC_PI_2 - minimum_angular_distance
            } else {
                -std::f32::consts::FRAC_PI_2 + minimum_angular_distance
            };
        }
        for bin in 0..number_of_bins {
            let frequency_squared = (bin as f32 * spectrum.fourier_voxel_size_y).powi(2);
            result.spatial_frequency[bin] = frequency_squared.sqrt() as f64;
            result.ctf_profile[bin] = ctf.evaluate(frequency_squared, azimuth);
            result.extrema_profile[bin] = ctf
                .number_of_extrema_before_squared_spatial_frequency(frequency_squared, azimuth)
                as f32;
        }
        let mut number_of_values = vec![0_usize; number_of_bins];
        for y in 0..spectrum.logical_y_dimension {
            for x in 0..spectrum.logical_x_dimension {
                let extrema_value = extrema.return_real_pixel_from_physical_coord(x, y, 0)?;
                let ctf_value = ctf_values.return_real_pixel_from_physical_coord(x, y, 0)?;
                let bin = return_spectrum_bin_number(
                    &result.extrema_profile,
                    extrema_value,
                    ctf_value,
                    &result.ctf_profile,
                )?;
                result.average[bin] +=
                    spectrum.return_real_pixel_from_physical_coord(x, y, 0)? as f64;
                number_of_values[bin] += 1;
            }
        }
        for bin in 0..number_of_bins {
            if number_of_values[bin] > 0 {
                result.average[bin] /= number_of_values[bin] as f64;
            }
            result.average_fit[bin] = result.ctf_profile[bin].abs() as f64;
        }
    }
    result.average_rank.clone_from(&result.average);
    renormalize_1d_spectrum_for_frc(
        &mut result.average_rank,
        &result.average_fit,
        &result.extrema_profile,
    );
    Some(result)
}

/// `FindRotationalAlignmentBetweenTwoStacksOfImages`.
pub fn find_rotational_alignment_between_two_stacks_of_images(
    images: &[Image],
    reference_images: &[Image],
    search_half_range: f32,
    search_step_size: f32,
    minimum_radius: f32,
    maximum_radius: f32,
) -> f32 {
    assert!(!images.is_empty() && images.len() == reference_images.len());
    assert!(search_step_size > 0.);
    let image = &images[0];
    let reference = &reference_images[0];
    assert!(image.is_in_real_space && reference.is_in_real_space);
    assert!(image.logical_z_dimension == 1 && reference.logical_z_dimension == 1);
    assert!(image.has_same_dimensions_as(reference));
    let minimum_radius_squared = minimum_radius.powi(2);
    let maximum_radius_squared = maximum_radius.powi(2);
    let inverse_x = 1. / image.logical_x_dimension as f32;
    let inverse_y = 1. / image.logical_y_dimension as f32;
    let mut best_correlation = -f32::MAX;
    let mut best_rotation = -f32::MAX;
    let mut rotation = -search_half_range;
    while rotation < search_half_range + search_step_size {
        let angle = rotation / 180. * std::f32::consts::PI;
        let mut numerator = EmpiricalDistribution::new();
        let mut self_denominator = EmpiricalDistribution::new();
        let mut reference_denominator = EmpiricalDistribution::new();
        for reference_image in reference_images {
            for y in 0..reference.logical_y_dimension {
                let y_logical = y - reference.physical_address_of_box_center_y;
                let y_radius = (y_logical as f32 * inverse_y).powi(2);
                for x in 0..reference.logical_x_dimension {
                    let x_logical = x - reference.physical_address_of_box_center_x;
                    let radius = (x_logical as f32 * inverse_x).powi(2) + y_radius;
                    if radius >= minimum_radius_squared && radius <= maximum_radius_squared {
                        let self_x = x_logical as f32 * angle.cos()
                            - y_logical as f32 * angle.sin()
                            + image.physical_address_of_box_center_x as f32;
                        let self_y = x_logical as f32 * angle.sin()
                            + y_logical as f32 * angle.cos()
                            + image.physical_address_of_box_center_y as f32;
                        if self_x as i32 > 0
                            && self_x as i32 + 1 < image.logical_x_dimension
                            && self_y as i32 > 0
                            && self_y as i32 + 1 < image.logical_y_dimension
                        {
                            let interpolated = image
                                .get_real_value_by_linear_interpolation_no_bounds_check_image(
                                    self_x, self_y,
                                );
                            let value = reference_image
                                .return_real_pixel_from_physical_coord(x, y, 0)
                                .unwrap();
                            numerator.add_sample_value(interpolated * value);
                            reference_denominator.add_sample_value(value.powi(2));
                            self_denominator.add_sample_value(interpolated.powi(2));
                        }
                    }
                }
            }
        }
        let correlation = numerator.get_sample_sum()
            / (reference_denominator.get_sample_sum() * self_denominator.get_sample_sum()).sqrt();
        if correlation > best_correlation {
            best_correlation = correlation;
            best_rotation = rotation;
        }
        rotation += search_step_size;
    }
    best_rotation
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curve_objective_and_owned_parameter_state_follow_source() {
        let ctf = Ctf::with_fitting_parameters(
            300., 2.7, 0.07, 15_000., 15_000., 0., 0.02, 0.45, -10., 1., 0.,
        );
        let comparison = CurveCtfComparison {
            curve: (0..64).map(|value| value as f32 + 1.).collect(),
            reciprocal_pixel_size: 1. / 64.,
            ctf,
            find_phase_shift: true,
        };
        assert!(ctffind_curve_objective_function(&comparison, &[15_000., 0.]).is_finite());
        assert_eq!(CtffindParams::default().box_size, 0);
    }

    #[test]
    fn frc_utilities_normalize_and_correlate_source_bins() {
        let mut average = vec![0., 1., 2., 1., 0., 1., 2., 1., 0.];
        let fit = average.clone();
        let extrema = vec![0., 0., 1., 1., 1., 2., 2., 2., 2.];
        renormalize_1d_spectrum_for_frc(&mut average, &fit, &extrema);
        let (frc, sigma) = compute_frc_between_1d_spectrum_and_fit(&average, &average, &extrema, 1);
        assert_eq!(frc[0], 1.);
        assert!(frc[3] > 0.99);
        assert!(sigma[3].is_finite());
    }

    #[test]
    fn ctf_overlay_zeros_low_frequency_and_writes_theoretical_quadrant() {
        let ctf = Ctf::with_fitting_parameters(
            300., 2.7, 0.07, 15_000., 15_000., 0., 0.1, 0.45, -10., 1., 0.,
        );
        let mut spectrum = Image::default();
        spectrum.allocate(32, 32, 1, true).unwrap();
        spectrum.set_to_constant(2.);
        overlay_ctf(&mut spectrum, &ctf);
        assert_eq!(
            spectrum.return_real_pixel_from_physical_coord(16, 16, 0),
            Some(0.)
        );
        assert_ne!(
            spectrum.return_real_pixel_from_physical_coord(6, 6, 0),
            Some(2.)
        );
    }

    #[test]
    fn rotational_rescaling_keeps_a_blank_or_flat_profile_safe() {
        let mut average = vec![0.; 8];
        let fit = vec![0.; 8];
        let frequencies = (0..8).map(|value| value as f64).collect::<Vec<_>>();
        assert!(rescale_rotational_average(
            &mut average,
            &fit,
            &frequencies,
            7
        ));
        assert_eq!(average, vec![0.; 8]);
    }

    #[test]
    fn extrema_images_and_bin_selection_follow_ctf_profiles() {
        let ctf = Ctf::with_fitting_parameters(
            300., 2.7, 0.07, 15_000., 15_000., 0., 0.02, 0.45, -10., 1., 0.,
        );
        let mut extrema = Image::default();
        let mut values = Image::default();
        extrema.allocate(16, 16, 1, true).unwrap();
        values.allocate(16, 16, 1, true).unwrap();
        compute_images_with_number_of_extrema_and_ctf_values(&ctf, &mut extrema, &mut values);
        assert_eq!(
            extrema.return_real_pixel_from_physical_coord(8, 8, 0),
            Some(0.)
        );
        assert_eq!(
            return_spectrum_bin_number(&[0., 1., 2.], 1., 0.7, &[0., 0.8, 0.]),
            Some(1)
        );
    }

    #[test]
    fn rotational_average_uses_ctf_extrema_to_collect_spectrum_values() {
        let ctf = Ctf::with_fitting_parameters(
            300., 2.7, 0.07, 15_000., 14_000., 0.2, 0.02, 0.45, -10., 1., 0.,
        );
        let mut spectrum = Image::default();
        let mut extrema = Image::default();
        let mut values = Image::default();
        spectrum.allocate(32, 32, 1, true).unwrap();
        extrema.allocate(32, 32, 1, true).unwrap();
        values.allocate(32, 32, 1, true).unwrap();
        for y in 0..32 {
            for x in 0..32 {
                spectrum.set_real_pixel_from_physical_coord(x, y, 0, (x + 2 * y) as f32);
            }
        }
        compute_images_with_number_of_extrema_and_ctf_values(&ctf, &mut extrema, &mut values);
        let result =
            compute_rotational_average_of_power_spectrum(&spectrum, &ctf, &extrema, &values, 16)
                .unwrap();
        assert_eq!(result.average.len(), 16);
        assert!(
            result
                .spatial_frequency
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
        );
        assert!(result.average.iter().any(|&value| value != 0.));
        assert!(result.average_fit.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rotational_alignment_selects_the_source_grid_angle() {
        let mut image = Image::default();
        let mut reference = Image::default();
        image.allocate(16, 16, 1, true).unwrap();
        reference.allocate(16, 16, 1, true).unwrap();
        for y in 0..16 {
            for x in 0..16 {
                let value = if x > 8 && y > 8 { 3. } else { 0. };
                image.set_real_pixel_from_physical_coord(x, y, 0, value);
                reference.set_real_pixel_from_physical_coord(x, y, 0, value);
            }
        }
        assert_eq!(
            find_rotational_alignment_between_two_stacks_of_images(
                &[image],
                &[reference],
                10.,
                5.,
                0.,
                1.
            ),
            0.
        );
    }

    #[test]
    fn exported_ctffind_runs_a_safe_owned_synthetic_spectrum_path() {
        let parameters = CtffindParams {
            pixel_size_of_input_image: 1.,
            acceleration_voltage: 300.,
            spherical_aberration: 2.7,
            amplitude_contrast: 0.07,
            box_size: 32,
            minimum_resolution: 20.,
            maximum_resolution: 4.,
            minimum_defocus: 15_000.,
            maximum_defocus: 15_000.,
            defocus_search_step: 1_000.,
            slower_search: false,
            astigmatism_tolerance: -10.,
            find_additional_phase_shift: false,
            minimum_additional_phase_shift: 0.,
            maximum_additional_phase_shift: 0.,
            additional_phase_shift_search_step: 0.1,
            astigmatism_is_known: true,
            known_astigmatism: 0.,
            known_astigmatism_angle: 0.,
            compute_extra_stats: false,
            noisy_input_image: true,
        };
        let spectrum = (0..32)
            .flat_map(|y| {
                (0..32).map(move |x| {
                    let dx = x as f32 - 16.;
                    let dy = y as f32 - 16.;
                    (dx * dx + dy * dy).sqrt() + ((13 * x + 7 * y) % 11) as f32
                })
            })
            .collect::<Vec<_>>();
        let result = ctffind(&parameters, &spectrum, 32).unwrap();
        assert_eq!(result.rotational_average.len(), 23);
        assert!(result.defocus_1.is_finite());
        assert!(result.normalized_average.is_none());
    }
}
