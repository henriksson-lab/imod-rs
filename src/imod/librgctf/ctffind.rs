//! Translation of `IMOD/librgctf/ctffind.cpp` and `IMOD/include/ctffind.h`.
//!
//! `ctffind.cpp` is the fitting function extracted from the ctffind program of
//! Rohou and Grigorieff; IMOD replaced `CtffindApp::DoInteractiveUserInput`
//! with the exported `ctffind()` entry point and removed the movie, gain
//! reference and file-output handling.  The `IMOD:` comments in the source mark
//! every such change and are reproduced here.

use super::brute_force_search::BruteForceSearch;
use super::conjugate_gradient::ConjugateGradient;
use super::ctf::Ctf;
use super::curve::Curve;
use super::defines::{ANSI_COLOR_RED, ANSI_COLOR_RESET, PI};
use super::empirical_distribution::EmpiricalDistribution;
use super::functions::{
    CharArgType, ctf_wall_time, internal_set_print_func, rank_sort, wx_printf, wx_printf_fmt,
};
use super::image::{Image, WriteSliceType, internal_set_write_slice_func};
use crate::imod::libcfshr::b3dutil::CArg;

/// C++ `const std::string ctffind_version = "4.1.9"` (`ctffind.cpp:13`).
pub const CTFFIND_VERSION: &str = "4.1.9";

/// C++ `struct CtffindParams` (`IMOD/include/ctffind.h:7`).
#[derive(Clone, Copy, Debug, PartialEq)]
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

impl Default for CtffindParams {
    /// The C++ struct has no constructor; a `CtffindParams` declared on the
    /// stack starts as whatever was there, and `testctffind.cpp` sets every
    /// member it uses.  These zeros are the Rust equivalent of that blank
    /// slate, not a set of defaults the library defines.
    fn default() -> Self {
        Self {
            pixel_size_of_input_image: 0.0,
            acceleration_voltage: 0.0,
            spherical_aberration: 0.0,
            amplitude_contrast: 0.0,
            box_size: 0,
            minimum_resolution: 0.0,
            maximum_resolution: 0.0,
            minimum_defocus: 0.0,
            maximum_defocus: 0.0,
            defocus_search_step: 0.0,
            slower_search: false,
            astigmatism_tolerance: 0.0,
            find_additional_phase_shift: false,
            minimum_additional_phase_shift: 0.0,
            maximum_additional_phase_shift: 0.0,
            additional_phase_shift_search_step: 0.0,
            astigmatism_is_known: false,
            known_astigmatism: 0.0,
            known_astigmatism_angle: 0.0,
            compute_extra_stats: false,
            noisy_input_image: false,
        }
    }
}

/// C++ `class ImageCTFComparison` (`ctffind.cpp:27`).
pub struct ImageCtfComparison {
    pub number_of_images: i32,
    /// Usually an amplitude spectrum, or an array of amplitude spectra
    pub img: Vec<Image>,
    pub number_to_correlate: i32,
    pub norm_image: f64,
    pub image_mean: f64,
    pub azimuths: Vec<f32>,
    pub spatial_frequency_squared: Vec<f32>,
    pub addresses: Vec<i32>,

    ctf: Ctf,
    pixel_size: f32,
    find_phase_shift: bool,
    astigmatism_is_known: bool,
    known_astigmatism: f32,
    known_astigmatism_angle: f32,
    fit_defocus_sweep: bool,
}

impl ImageCtfComparison {
    /// C++ `ImageCTFComparison::ImageCTFComparison` (`ctffind.cpp:67`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wanted_number_of_images: i32,
        wanted_ctf: Ctf,
        wanted_pixel_size: f32,
        should_find_phase_shift: bool,
        wanted_astigmatism_is_known: bool,
        wanted_known_astigmatism: f32,
        wanted_known_astigmatism_angle: f32,
        should_fit_defocus_sweep: bool,
    ) -> Self {
        Self {
            number_of_images: wanted_number_of_images,
            img: (0..wanted_number_of_images).map(|_| Image::new()).collect(),
            ctf: wanted_ctf,
            pixel_size: wanted_pixel_size,
            find_phase_shift: should_find_phase_shift,
            astigmatism_is_known: wanted_astigmatism_is_known,
            known_astigmatism: wanted_known_astigmatism,
            known_astigmatism_angle: wanted_known_astigmatism_angle,
            fit_defocus_sweep: should_fit_defocus_sweep,
            azimuths: Vec::new(),
            spatial_frequency_squared: Vec::new(),
            addresses: Vec::new(),
            number_to_correlate: 0,
            norm_image: 0.0,
            image_mean: 0.0,
        }
    }

    /// C++ `ImageCTFComparison::SetImage` (`ctffind.cpp:98`).
    pub fn set_image(&mut self, wanted_image_number: i32, new_image: &Image) {
        self.img[wanted_image_number as usize].copy_from(new_image);
    }

    /// C++ `ImageCTFComparison::SetCTF` (`ctffind.cpp:104`).
    pub fn set_ctf(&mut self, new_ctf: Ctf) {
        self.ctf = new_ctf;
    }

    /// C++ `ImageCTFComparison::SetupQuickCorrelation` (`ctffind.cpp:109`).
    pub fn setup_quick_correlation(&mut self) {
        let mut number_to_correlate = 0;
        let mut norm_image = 0.0;
        let mut image_mean = 0.0;
        self.img[0].setup_quick_correlation_with_ctf(
            self.ctf,
            &mut number_to_correlate,
            &mut norm_image,
            &mut image_mean,
            None,
            None,
            None,
        );
        self.number_to_correlate = number_to_correlate;
        self.norm_image = norm_image;
        self.image_mean = image_mean;
        self.azimuths = vec![0.0; self.number_to_correlate as usize];
        self.spatial_frequency_squared = vec![0.0; self.number_to_correlate as usize];
        self.addresses = vec![0; self.number_to_correlate as usize];
        let mut addresses = std::mem::take(&mut self.addresses);
        let mut spatial_frequency_squared = std::mem::take(&mut self.spatial_frequency_squared);
        let mut azimuths = std::mem::take(&mut self.azimuths);
        self.img[0].setup_quick_correlation_with_ctf(
            self.ctf,
            &mut number_to_correlate,
            &mut norm_image,
            &mut image_mean,
            Some(&mut addresses),
            Some(&mut spatial_frequency_squared),
            Some(&mut azimuths),
        );
        self.addresses = addresses;
        self.spatial_frequency_squared = spatial_frequency_squared;
        self.azimuths = azimuths;
        self.number_to_correlate = number_to_correlate;
        self.norm_image = norm_image;
        self.image_mean = image_mean;
    }

    /// C++ `ImageCTFComparison::ReturnCTF` (`ctffind.cpp:118`).
    pub fn return_ctf(&self) -> Ctf {
        self.ctf
    }

    /// C++ `ImageCTFComparison::AstigmatismIsKnown` (`ctffind.cpp:119`).
    pub fn astigmatism_is_known(&self) -> bool {
        self.astigmatism_is_known
    }

    /// C++ `ImageCTFComparison::ReturnKnownAstigmatism` (`ctffind.cpp:120`).
    pub fn return_known_astigmatism(&self) -> f32 {
        self.known_astigmatism
    }

    /// C++ `ImageCTFComparison::ReturnKnownAstigmatismAngle` (`ctffind.cpp:121`).
    pub fn return_known_astigmatism_angle(&self) -> f32 {
        self.known_astigmatism_angle
    }

    /// C++ `ImageCTFComparison::FindPhaseShift` (`ctffind.cpp:122`).
    pub fn find_phase_shift(&self) -> bool {
        self.find_phase_shift
    }

    /// The private `pixel_size` member (`ctffind.cpp:50`), which the class
    /// stores and never reads.
    pub fn pixel_size(&self) -> f32 {
        self.pixel_size
    }

    /// The private `fit_defocus_sweep` member (`ctffind.cpp:55`), likewise.
    pub fn fit_defocus_sweep(&self) -> bool {
        self.fit_defocus_sweep
    }
}

/// C++ `class CurveCTFComparison` (`ctffind.cpp:58`).
pub struct CurveCtfComparison {
    /// Usually the 1D rotational average of the amplitude spectrum of an image
    pub curve: Vec<f32>,
    pub number_of_bins: i32,
    /// In reciprocal pixels
    pub reciprocal_pixel_size: f32,
    pub ctf: Ctf,
    pub find_phase_shift: bool,
}

impl Default for CurveCtfComparison {
    fn default() -> Self {
        Self {
            curve: Vec::new(),
            number_of_bins: 0,
            reciprocal_pixel_size: 0.0,
            ctf: Ctf::new(),
            find_phase_shift: false,
        }
    }
}

/// C++ `static bool inConjGrad = false` (`ctffind.cpp:126`).
///
/// The source only reads it from a commented-out debug print.
static IN_CONJ_GRAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// C++ `CtffindObjectiveFunction` (`ctffind.cpp:129`): the function which will
/// be minimised.
pub fn ctffind_objective_function(
    comparison_object: &ImageCtfComparison,
    array_of_values: &[f32],
) -> f32 {
    let mut my_ctf = comparison_object.return_ctf();
    if comparison_object.astigmatism_is_known() {
        my_ctf.set_defocus(
            array_of_values[0],
            array_of_values[0] - comparison_object.return_known_astigmatism(),
            comparison_object.return_known_astigmatism_angle(),
        );
    } else {
        my_ctf.set_defocus(array_of_values[0], array_of_values[1], array_of_values[2]);
    }
    if comparison_object.find_phase_shift() {
        if comparison_object.astigmatism_is_known() {
            my_ctf.set_additional_phase_shift(array_of_values[1]);
        } else {
            my_ctf.set_additional_phase_shift(array_of_values[3]);
        }
    }

    // Evaluate the function
    if comparison_object.number_to_correlate != 0 {
        -comparison_object.img[0].quick_correlation_with_ctf(
            my_ctf,
            comparison_object.number_to_correlate,
            comparison_object.norm_image,
            comparison_object.image_mean,
            &comparison_object.addresses,
            &comparison_object.spatial_frequency_squared,
            &comparison_object.azimuths,
        )
    } else {
        -comparison_object.img[0].get_correlation_with_ctf(my_ctf)
    }
}

/// C++ `CtffindCurveObjectiveFunction` (`ctffind.cpp:163`): the function which
/// will be minimised when dealing with 1D fitting.
pub fn ctffind_curve_objective_function(
    comparison_object: &CurveCtfComparison,
    array_of_values: &[f32],
) -> f32 {
    let mut my_ctf = comparison_object.ctf;
    my_ctf.set_defocus(array_of_values[0], array_of_values[0], 0.0);
    if comparison_object.find_phase_shift {
        my_ctf.set_additional_phase_shift(array_of_values[1]);
    }

    // Compute the cross-correlation
    let mut cross_product = 0.0f64;
    let mut norm_curve = 0.0f64;
    let mut norm_ctf = 0.0f64;
    let mut number_of_values = 0i32;
    let mut current_spatial_frequency_squared: f32;
    let lowest_freq = f64::from(my_ctf.get_lowest_frequency_for_fitting()).powi(2) as f32;
    let highest_freq = f64::from(my_ctf.get_highest_frequency_for_fitting()).powi(2) as f32;
    let mut current_ctf_value: f32;

    for bin_counter in 0..comparison_object.number_of_bins as usize {
        current_spatial_frequency_squared =
            f64::from((bin_counter as f32) * comparison_object.reciprocal_pixel_size).powi(2)
                as f32;
        if current_spatial_frequency_squared > lowest_freq
            && current_spatial_frequency_squared < highest_freq
        {
            current_ctf_value = my_ctf
                .evaluate(current_spatial_frequency_squared, 0.0)
                .abs();
            number_of_values += 1;
            cross_product +=
                f64::from(comparison_object.curve[bin_counter]) * f64::from(current_ctf_value);
            norm_curve += f64::from(comparison_object.curve[bin_counter]).powi(2);
            norm_ctf += f64::from(current_ctf_value).powi(2);
        }
    }
    let _ = number_of_values;

    // Note, we are not properly normalizing the cross correlation coefficient.
    (-cross_product / f64::from(((norm_ctf * norm_curve) as f32).sqrt())) as f32
}

/// C++ `ctffindSetSliceWriteFunc` (`ctffind.cpp:232`).
pub fn ctffind_set_slice_write_func(func: Option<WriteSliceType>) {
    internal_set_write_slice_func(func);
}

/// C++ `ctffindSetPrintFunc` (`ctffind.cpp:237`).
pub fn ctffind_set_print_func(func: Option<CharArgType>) {
    internal_set_print_func(func);
}

/// C++ `ctffind` (`ctffind.cpp:246`), the exported entry point.
///
/// `rotational_avg_out`, `normalized_avg_out` and `fit_curve_out` stand in for
/// the C's `float **` out-parameters: `None` is the source's `NULL`, and a
/// `Some` receives the vector the C would have `malloc`ed.
#[allow(clippy::too_many_arguments)]
pub fn ctffind(
    params: &CtffindParams,
    spectrum_array: &[f32],
    nx_dim_in: i32,
    results_array: &mut [f32],
    mut rotational_avg_out: Option<&mut Option<Vec<f32>>>,
    mut normalized_avg_out: Option<&mut Option<Vec<f32>>>,
    mut fit_curve_out: Option<&mut Option<Vec<f32>>>,
    num_points_out: &mut i32,
    last_bin_freq_out: &mut f32,
) -> bool {
    // Arguments for this job
    let pixel_size_of_input_image = params.pixel_size_of_input_image;
    let _ = pixel_size_of_input_image;
    let acceleration_voltage = params.acceleration_voltage;
    let spherical_aberration = params.spherical_aberration;
    let amplitude_contrast = params.amplitude_contrast;
    let box_size = params.box_size;
    let minimum_resolution = params.minimum_resolution;
    let maximum_resolution = params.maximum_resolution;
    let minimum_defocus = params.minimum_defocus;
    let maximum_defocus = params.maximum_defocus;
    let defocus_search_step = params.defocus_search_step;
    let slower_search = params.slower_search;
    let astigmatism_tolerance = params.astigmatism_tolerance;
    let find_additional_phase_shift = params.find_additional_phase_shift;
    let minimum_additional_phase_shift = params.minimum_additional_phase_shift;
    let maximum_additional_phase_shift = params.maximum_additional_phase_shift;
    let additional_phase_shift_search_step = params.additional_phase_shift_search_step;
    let astigmatism_is_known = params.astigmatism_is_known;
    let known_astigmatism = params.known_astigmatism;
    let known_astigmatism_angle = params.known_astigmatism_angle;
    let compute_extra_stats = params.compute_extra_stats;

    // IMOD 8/3/23: something like this is being added to avoid CosineMask
    let noisy_input_image = params.noisy_input_image;

    let pixel_size_for_fitting = params.pixel_size_of_input_image;

    // Maybe the user wants to hold the phase shift value (which they can do by
    // giving the same value for min and max)
    let fixed_additional_phase_shift =
        (maximum_additional_phase_shift - minimum_additional_phase_shift).abs() < 0.01;

    // This could become a user-supplied parameter later
    let follow_1d_search_with_local_2d_brute_force = false;

    // Other variables
    let mut average_spectrum = Image::new();
    let mut average_spectrum_masked = Image::new();
    let mut current_power_spectrum = Image::new();
    let mut temp_image = Image::new();
    let mut current_ctf = Ctf::new();
    let mut average = 0.0f32;
    let mut sigma = 0.0f32;
    let mut convolution_box_size: i32;
    let mut comparison_object_1d = CurveCtfComparison::default();
    let estimated_astigmatism_angle: f32;
    // The source declares these as `float [4]` on the stack, so slots it never
    // writes hold whatever was there; see the note on `bf_halfrange[3]` below.
    let mut bf_halfrange = [0.0f32; 4];
    let mut bf_midpoint = [0.0f32; 4];
    let mut bf_stepsize = [0.0f32; 4];
    let mut cg_starting_point = [0.0f32; 4];
    let mut cg_accuracy = [0.0f32; 4];
    let mut number_of_search_dimensions: i32;
    let mut number_of_bins_in_1d_spectra: i32;
    let mut number_of_averaged_pixels = Curve::new();
    let mut rotational_average = Curve::new();
    let mut number_of_extrema_image = Image::new();
    let mut ctf_values_image = Image::new();
    let mut rotational_average_astig: Vec<f64> = Vec::new();
    let mut rotational_average_astig_renormalized: Vec<f64> = Vec::new();
    let mut spatial_frequency: Vec<f64> = Vec::new();
    let mut rotational_average_astig_fit: Vec<f64> = Vec::new();
    let mut number_of_extrema_profile: Vec<f32> = Vec::new();
    let mut ctf_values_profile: Vec<f32> = Vec::new();
    let mut fit_frc: Vec<f64> = Vec::new();
    let mut fit_frc_sigma: Vec<f64> = Vec::new();
    let mut last_bin_with_good_fit: i32;
    let mut best_score_after_initial_phase: f32 = 0.0;
    let mut last_bin_without_aliasing: i32 = 0;
    let intermediate_resolution = 5.0f32;

    // IMOD: set these variables
    let is_running_locally = true;

    // IMOD: print the messages
    // Some argument checking
    if minimum_resolution < maximum_resolution {
        wx_printf_fmt(
            "Error: Minimum resolution (%f) higher than maximum resolution (%f).",
            &[
                CArg::Dbl(f64::from(minimum_resolution)),
                CArg::Dbl(f64::from(maximum_resolution)),
            ],
        );
        return false;
    }
    if minimum_defocus > maximum_defocus {
        wx_printf("Error: Minimum defocus must be less than maximum defocus.");
        return false;
    }

    // IMOD: remove setting up loops, preparation of output files, gain reference
    let wall_start = ctf_wall_time();
    let _ = wall_start;

    // Prepare the average spectrum image
    average_spectrum.allocate_2d(box_size, box_size, true);

    // IMOD: Copy the input spectrum into the image array
    for iy in 0..box_size {
        let dst = (iy * (box_size + average_spectrum.padding_jump_value)) as usize;
        let src = (iy * nx_dim_in) as usize;
        average_spectrum.real_values[dst..dst + box_size as usize]
            .copy_from_slice(&spectrum_array[src..src + box_size as usize]);
    }

    // IMOD: remove loop on input images, change this from current_power_spectrum
    // to average_spectrum
    // Set origin of amplitude spectrum to 0.0
    let origin = average_spectrum.return_real_1d_address_from_physical_coord(
        average_spectrum.physical_address_of_box_center_x,
        average_spectrum.physical_address_of_box_center_y,
        average_spectrum.physical_address_of_box_center_z,
    );
    average_spectrum.real_values[origin as usize] = 0.0;

    // Filter the amplitude spectrum, remove background
    // IMOD: do this unconditionally
    // Try to weaken cross artefacts
    average_spectrum.compute_average_and_sigma_of_values_in_spectrum(
        (average_spectrum.logical_x_dimension as f32) * pixel_size_for_fitting / minimum_resolution,
        average_spectrum.logical_x_dimension as f32,
        &mut average,
        &mut sigma,
        12,
    );
    average_spectrum.divide_by_constant(sigma);
    average_spectrum.set_maximum_value_on_central_cross((f64::from(average / sigma) + 10.0) as f32);

    // Compute low-pass filtered version of the spectrum
    convolution_box_size = (f64::from(
        (average_spectrum.logical_x_dimension as f32) * pixel_size_for_fitting / minimum_resolution,
    ) * 2.0f64.sqrt()) as i32;
    if super::functions::is_even(convolution_box_size) {
        convolution_box_size += 1;
    }
    current_power_spectrum.allocate_2d(
        average_spectrum.logical_x_dimension,
        average_spectrum.logical_y_dimension,
        true,
    );
    // According to valgrind, this avoids potential problems later on.
    current_power_spectrum.set_to_constant(0.0);
    average_spectrum.spectrum_box_convolution(
        &mut current_power_spectrum,
        convolution_box_size,
        (average_spectrum.logical_x_dimension as f32) * pixel_size_for_fitting / minimum_resolution,
    );

    // Subtract low-pass-filtered spectrum from the spectrum.  This should
    // remove the background slope.
    average_spectrum.subtract_image(&current_power_spectrum);

    // Threshold high values
    let maximum = average_spectrum.return_maximum_value(3.0, 3.0);
    average_spectrum.set_maximum_value(maximum);

    average_spectrum_masked.copy_from(&average_spectrum);

    // IMOD 8/3/23
    if !noisy_input_image {
        average_spectrum_masked.cosine_mask(
            (average_spectrum_masked.logical_x_dimension as f32) * pixel_size_for_fitting
                / if maximum_resolution < 8.0f32 {
                    8.0f32
                } else {
                    maximum_resolution
                },
            (average_spectrum_masked.logical_x_dimension as f32) * pixel_size_for_fitting
                / if maximum_resolution < 4.0f32 {
                    4.0f32
                } else {
                    maximum_resolution
                },
            true,
            false,
            0.0,
        );
    }

    /*
     * We now have a spectrum which we can use to fit CTFs
     */

    // Set up the CTF object
    current_ctf.init_with_fitting_parameters(
        acceleration_voltage,
        spherical_aberration,
        amplitude_contrast,
        minimum_defocus,
        minimum_defocus,
        0.0,
        (1.0 / f64::from(minimum_resolution)) as f32,
        (1.0 / f64::from(if maximum_resolution < intermediate_resolution {
            intermediate_resolution
        } else {
            maximum_resolution
        })) as f32,
        astigmatism_tolerance,
        pixel_size_for_fitting,
        minimum_additional_phase_shift,
    );
    current_ctf.set_defocus(
        minimum_defocus / pixel_size_for_fitting,
        minimum_defocus / pixel_size_for_fitting,
        0.0,
    );
    current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);

    // Set up the comparison object
    // DNM: Do not tell it to find phase if it is fixed
    let mut comparison_object_2d = ImageCtfComparison::new(
        1,
        current_ctf,
        pixel_size_for_fitting,
        find_additional_phase_shift && !fixed_additional_phase_shift,
        astigmatism_is_known,
        known_astigmatism / pixel_size_for_fitting,
        (f64::from(known_astigmatism_angle) / 180.0 * PI) as f32,
        false,
    );
    comparison_object_2d.set_image(0, &average_spectrum_masked);
    comparison_object_2d.setup_quick_correlation();

    // Let's look for the astigmatism angle first
    if astigmatism_is_known {
        estimated_astigmatism_angle = known_astigmatism_angle;
    } else {
        temp_image.copy_from(&average_spectrum);
        temp_image.apply_mirror_along_y();
        estimated_astigmatism_angle = (0.5
            * f64::from(find_rotational_alignment_between_two_stacks_of_images(
                &average_spectrum,
                &temp_image,
                1,
                90.0,
                5.0,
                pixel_size_for_fitting / minimum_resolution,
                pixel_size_for_fitting
                    / if maximum_resolution < intermediate_resolution {
                        intermediate_resolution
                    } else {
                        maximum_resolution
                    },
            ))) as f32;
    }

    /*
     * Initial brute-force search, in 1D (fast, but not as accurate)
     */
    if !slower_search {
        // 1D rotational average
        number_of_bins_in_1d_spectra = average_spectrum_masked
            .return_maximum_diagonal_radius()
            .ceil() as i32;
        rotational_average.setup_x_axis(
            0.0,
            (number_of_bins_in_1d_spectra as f32) * average_spectrum_masked.fourier_voxel_size_x,
            number_of_bins_in_1d_spectra,
        );
        number_of_averaged_pixels.assign(&rotational_average);
        average_spectrum_masked.compute_1d_rotational_average(
            &mut rotational_average,
            &mut number_of_averaged_pixels,
            true,
        );

        comparison_object_1d.ctf = current_ctf;
        comparison_object_1d.curve = vec![0.0; number_of_bins_in_1d_spectra as usize];
        for counter in 0..number_of_bins_in_1d_spectra as usize {
            comparison_object_1d.curve[counter] = rotational_average.data_y[counter];
        }

        // DNM: Do not find phase if it is fixed
        comparison_object_1d.find_phase_shift =
            find_additional_phase_shift && !fixed_additional_phase_shift;
        comparison_object_1d.number_of_bins = number_of_bins_in_1d_spectra;
        comparison_object_1d.reciprocal_pixel_size = average_spectrum_masked.fourier_voxel_size_x;

        // We can now look for the defocus value
        bf_halfrange[0] = (0.5 * f64::from(maximum_defocus - minimum_defocus)
            / f64::from(pixel_size_for_fitting)) as f32;
        bf_halfrange[1] = (0.5
            * f64::from(maximum_additional_phase_shift - minimum_additional_phase_shift))
            as f32;

        bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
        bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[1];

        bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
        bf_stepsize[1] = additional_phase_shift_search_step;

        if find_additional_phase_shift && !fixed_additional_phase_shift {
            number_of_search_dimensions = 2;
        } else {
            number_of_search_dimensions = 1;
        }

        // Actually run the BF search
        let mut brute_force_search = BruteForceSearch::new();
        brute_force_search.init(
            |values| ctffind_curve_objective_function(&comparison_object_1d, values),
            number_of_search_dimensions,
            &bf_midpoint,
            &bf_halfrange,
            &bf_stepsize,
            false,
            false,
        );
        brute_force_search.run();

        // We can now do a local optimization
        // The end point of the BF search is the beginning of the CG search
        for counter in 0..number_of_search_dimensions as usize {
            cg_starting_point[counter] = brute_force_search.get_best_value(counter as i32);
        }
        cg_accuracy[0] = 100.0;
        cg_accuracy[1] = 0.05;
        let mut conjugate_gradient_minimizer = ConjugateGradient::new();
        conjugate_gradient_minimizer.init(
            |values| ctffind_curve_objective_function(&comparison_object_1d, values),
            number_of_search_dimensions,
            &cg_starting_point,
            &cg_accuracy,
        );
        conjugate_gradient_minimizer.run();
        for counter in 0..number_of_search_dimensions as usize {
            cg_starting_point[counter] =
                conjugate_gradient_minimizer.get_best_value(counter as i32);
        }
        current_ctf.set_defocus(
            cg_starting_point[0],
            cg_starting_point[0],
            (f64::from(estimated_astigmatism_angle) / 180.0 * PI) as f32,
        );
        if find_additional_phase_shift {
            if fixed_additional_phase_shift {
                current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);
            } else {
                current_ctf.set_additional_phase_shift(cg_starting_point[1]);
            }
        }

        // Remember the best score so far
        best_score_after_initial_phase = -conjugate_gradient_minimizer.get_best_score();
    } // end of the fast search over the 1D function

    /*
     * Brute-force search over the 2D scoring function.
     */
    if slower_search || (!slower_search && follow_1d_search_with_local_2d_brute_force) {
        // Setup the parameters for the brute force search

        if slower_search {
            // This is the first search we are doing - scan the entire range
            if astigmatism_is_known {
                bf_halfrange[0] = (0.5 * f64::from(maximum_defocus - minimum_defocus)
                    / f64::from(pixel_size_for_fitting)) as f32;
                bf_halfrange[1] = (0.5
                    * f64::from(maximum_additional_phase_shift - minimum_additional_phase_shift))
                    as f32;

                bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                // NOTE: the source really does read `bf_halfrange[3]` here,
                // which nothing has written on this path — an uninitialised
                // stack read in the C.  A Rust array is zeroed, so this is
                // `minimum_additional_phase_shift + 0`.
                bf_midpoint[1] = minimum_additional_phase_shift + bf_halfrange[3];

                bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                bf_stepsize[1] = additional_phase_shift_search_step;

                if find_additional_phase_shift && !fixed_additional_phase_shift {
                    number_of_search_dimensions = 2;
                } else {
                    number_of_search_dimensions = 1;
                }
            } else {
                bf_halfrange[0] = (0.5 * f64::from(maximum_defocus - minimum_defocus)
                    / f64::from(pixel_size_for_fitting)) as f32;
                bf_halfrange[1] = bf_halfrange[0];
                bf_halfrange[2] = 0.0;
                bf_halfrange[3] = (0.5
                    * f64::from(maximum_additional_phase_shift - minimum_additional_phase_shift))
                    as f32;

                bf_midpoint[0] = minimum_defocus / pixel_size_for_fitting + bf_halfrange[0];
                bf_midpoint[1] = bf_midpoint[0];
                bf_midpoint[2] = (f64::from(estimated_astigmatism_angle) / 180.0 * PI) as f32;
                bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

                bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                bf_stepsize[1] = bf_stepsize[0];
                bf_stepsize[2] = 0.0;
                bf_stepsize[3] = additional_phase_shift_search_step;

                if find_additional_phase_shift && !fixed_additional_phase_shift {
                    number_of_search_dimensions = 4;
                } else {
                    number_of_search_dimensions = 3;
                }
            }
        } else {
            // we will do a brute-force search near the result of the search
            // over the 1D objective function
            if astigmatism_is_known {
                bf_midpoint[0] = current_ctf.get_defocus_1();
                bf_midpoint[1] = current_ctf.get_additional_phase_shift();

                bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                bf_stepsize[1] = additional_phase_shift_search_step;

                bf_halfrange[0] = (2.0 * f64::from(defocus_search_step)
                    / f64::from(pixel_size_for_fitting)
                    + 0.1) as f32;
                bf_halfrange[1] =
                    (2.0 * f64::from(additional_phase_shift_search_step) + 0.01) as f32;

                if find_additional_phase_shift && !fixed_additional_phase_shift {
                    number_of_search_dimensions = 2;
                } else {
                    number_of_search_dimensions = 1;
                }
            } else {
                bf_midpoint[0] = current_ctf.get_defocus_1();
                bf_midpoint[1] = current_ctf.get_defocus_2();
                bf_midpoint[2] = current_ctf.get_astigmatism_azimuth();
                // Same uninitialised `bf_halfrange[3]` read as above.
                bf_midpoint[3] = minimum_additional_phase_shift + bf_halfrange[3];

                bf_stepsize[0] = defocus_search_step / pixel_size_for_fitting;
                bf_stepsize[1] = bf_stepsize[0];
                bf_stepsize[2] = 0.0;
                bf_stepsize[3] = additional_phase_shift_search_step;

                if astigmatism_tolerance > 0.0 {
                    bf_halfrange[0] = (2.0 * f64::from(astigmatism_tolerance)
                        / f64::from(pixel_size_for_fitting)
                        + 0.1) as f32;
                } else {
                    bf_halfrange[0] = (2.0 * f64::from(defocus_search_step)
                        / f64::from(pixel_size_for_fitting)
                        + 0.1) as f32;
                }
                bf_halfrange[1] = bf_halfrange[0];
                bf_halfrange[2] = 0.0;
                bf_halfrange[3] =
                    (2.0 * f64::from(additional_phase_shift_search_step) + 0.01) as f32;

                if find_additional_phase_shift && !fixed_additional_phase_shift {
                    number_of_search_dimensions = 4;
                } else {
                    number_of_search_dimensions = 3;
                }
            }
        }

        // DNM: Do one-time set of phase shift for fixed value
        if find_additional_phase_shift && fixed_additional_phase_shift {
            current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);
        }

        // Actually run the BF search (we run a local minimizer at every grid
        // point only if this is a refinement search following 1D search)
        let mut brute_force_search = BruteForceSearch::new();
        brute_force_search.init(
            |values| ctffind_objective_function(&comparison_object_2d, values),
            number_of_search_dimensions,
            &bf_midpoint,
            &bf_halfrange,
            &bf_stepsize,
            !slower_search,
            is_running_locally,
        );
        brute_force_search.run();

        // The end point of the BF search is the beginning of the CG search
        for counter in 0..number_of_search_dimensions as usize {
            cg_starting_point[counter] = brute_force_search.get_best_value(counter as i32);
        }

        if astigmatism_is_known {
            current_ctf.set_defocus(
                cg_starting_point[0],
                cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting,
                (f64::from(known_astigmatism_angle) / 180.0 * PI) as f32,
            );
            if find_additional_phase_shift {
                if fixed_additional_phase_shift {
                    current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);
                } else {
                    current_ctf.set_additional_phase_shift(cg_starting_point[1]);
                }
            }
        } else {
            current_ctf.set_defocus(
                cg_starting_point[0],
                cg_starting_point[1],
                cg_starting_point[2],
            );
            if find_additional_phase_shift {
                if fixed_additional_phase_shift {
                    current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);
                } else {
                    current_ctf.set_additional_phase_shift(cg_starting_point[3]);
                }
            }
        }

        current_ctf.enforce_convention();

        // Remember the best score so far
        best_score_after_initial_phase = -brute_force_search.get_best_score();
    }
    let _ = best_score_after_initial_phase;

    /*
     * Set up the conjugate gradient minimization of the 2D scoring function
     */
    if astigmatism_is_known {
        cg_starting_point[0] = current_ctf.get_defocus_1();
        if find_additional_phase_shift {
            cg_starting_point[1] = current_ctf.get_additional_phase_shift();
        }
        if find_additional_phase_shift && !fixed_additional_phase_shift {
            number_of_search_dimensions = 2;
        } else {
            number_of_search_dimensions = 1;
        }
        cg_accuracy[0] = 100.0;
        cg_accuracy[1] = 0.05;
    } else {
        cg_accuracy[0] = 100.0;
        cg_accuracy[1] = 100.0;
        cg_accuracy[2] = 0.025;
        cg_accuracy[3] = 0.05;
        cg_starting_point[0] = current_ctf.get_defocus_1();
        cg_starting_point[1] = current_ctf.get_defocus_2();
        if slower_search || (!slower_search && follow_1d_search_with_local_2d_brute_force) {
            // we did a search against the 2D power spectrum so we have a better
            // estimate of the astigmatism angle in the CTF object
            cg_starting_point[2] = current_ctf.get_astigmatism_azimuth();
        } else {
            // all we have right now is the guessed astigmatism angle from the
            // mirror trick before any CTF fitting was even tried
            cg_starting_point[2] = (f64::from(estimated_astigmatism_angle) / 180.0 * PI) as f32;
        }

        if find_additional_phase_shift {
            cg_starting_point[3] = current_ctf.get_additional_phase_shift();
        }
        if find_additional_phase_shift && !fixed_additional_phase_shift {
            number_of_search_dimensions = 4;
        } else {
            number_of_search_dimensions = 3;
        }
    }
    // CG minimization
    IN_CONJ_GRAD.store(true, std::sync::atomic::Ordering::Relaxed);
    comparison_object_2d.set_ctf(current_ctf);
    let final_best_score;
    {
        let mut conjugate_gradient_minimizer = ConjugateGradient::new();
        conjugate_gradient_minimizer.init(
            |values| ctffind_objective_function(&comparison_object_2d, values),
            number_of_search_dimensions,
            &cg_starting_point,
            &cg_accuracy,
        );
        current_ctf.init_with_fitting_parameters(
            acceleration_voltage,
            spherical_aberration,
            amplitude_contrast,
            minimum_defocus,
            minimum_defocus,
            0.0,
            (1.0 / f64::from(minimum_resolution)) as f32,
            (1.0 / f64::from(maximum_resolution)) as f32,
            astigmatism_tolerance,
            pixel_size_for_fitting,
            minimum_additional_phase_shift,
        );
        conjugate_gradient_minimizer.run();

        // Remember the results of the refinement
        for counter in 0..number_of_search_dimensions as usize {
            cg_starting_point[counter] =
                conjugate_gradient_minimizer.get_best_value(counter as i32);
        }
        final_best_score = conjugate_gradient_minimizer.get_best_score();
    }
    if astigmatism_is_known {
        current_ctf.set_defocus(
            cg_starting_point[0],
            cg_starting_point[0] - known_astigmatism / pixel_size_for_fitting,
            (f64::from(known_astigmatism_angle) / 180.0 * PI) as f32,
        );
        if find_additional_phase_shift {
            if fixed_additional_phase_shift {
                current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);
            } else {
                current_ctf.set_additional_phase_shift(cg_starting_point[1]);
            }
        }
    } else {
        current_ctf.set_defocus(
            cg_starting_point[0],
            cg_starting_point[1],
            cg_starting_point[2],
        );
        if find_additional_phase_shift {
            if fixed_additional_phase_shift {
                current_ctf.set_additional_phase_shift(minimum_additional_phase_shift);
            } else {
                current_ctf.set_additional_phase_shift(cg_starting_point[3]);
            }
        }
    }
    current_ctf.enforce_convention();

    // Generate diagnostic image
    let edges = average_spectrum.return_average_of_real_values_on_edges();
    average_spectrum.add_constant((-1.0 * f64::from(edges)) as f32);

    /*
     *  Attempt some renormalisations - we want to do this over a range not
     *  affected by the central peak or strong Thon rings
     */
    let start_zero = current_ctf
        .return_squared_spatial_frequency_of_a_zero(3, 0.0)
        .sqrt();
    let finish_zero = current_ctf
        .return_squared_spatial_frequency_of_a_zero(4, 0.0)
        .sqrt();
    let mut normalization_radius_min = start_zero * (average_spectrum.logical_x_dimension as f32);
    let mut normalization_radius_max = finish_zero * (average_spectrum.logical_x_dimension as f32);

    if start_zero > current_ctf.get_highest_frequency_for_fitting()
        || start_zero < current_ctf.get_lowest_frequency_for_fitting()
        || finish_zero > current_ctf.get_highest_frequency_for_fitting()
        || finish_zero < current_ctf.get_lowest_frequency_for_fitting()
    {
        normalization_radius_max = current_ctf.get_highest_frequency_for_fitting()
            * (average_spectrum.logical_x_dimension as f32);
        let candidate = current_ctf.get_lowest_frequency_for_fitting()
            * (average_spectrum.logical_x_dimension as f32);
        let half = 0.5f32 * normalization_radius_max;
        normalization_radius_min = if half < candidate { candidate } else { half };
    }

    if normalization_radius_max - normalization_radius_min > 2.0 {
        average_spectrum.compute_average_and_sigma_of_values_in_spectrum(
            normalization_radius_min,
            normalization_radius_max,
            &mut average,
            &mut sigma,
            2,
        );
        average_spectrum.circle_mask(5.0, true);
        average_spectrum.set_maximum_value_on_central_cross(average);
        average_spectrum.set_minimum_and_maximum_values(
            (f64::from(average) - 4.0 * f64::from(sigma)) as f32,
            (f64::from(average) + 4.0 * f64::from(sigma)) as f32,
        );
        average_spectrum.compute_average_and_sigma_of_values_in_spectrum(
            normalization_radius_min,
            normalization_radius_max,
            &mut average,
            &mut sigma,
            2,
        );
        average_spectrum.add_constant((-1.0 * f64::from(average)) as f32);
        average_spectrum.multiply_by_constant((1.0 / f64::from(sigma)) as f32);
        average_spectrum.add_constant(average);
    }

    // 1D rotational average
    number_of_bins_in_1d_spectra = average_spectrum.return_maximum_diagonal_radius().ceil() as i32;
    rotational_average.setup_x_axis(
        0.0,
        (number_of_bins_in_1d_spectra as f32) * average_spectrum.fourier_voxel_size_x,
        number_of_bins_in_1d_spectra,
    );
    rotational_average.zero_y_data();
    number_of_averaged_pixels.assign(&rotational_average);
    average_spectrum.compute_1d_rotational_average(
        &mut rotational_average,
        &mut number_of_averaged_pixels,
        true,
    );

    // Rotational average, taking astigmatism into account
    if compute_extra_stats {
        number_of_extrema_image.allocate_2d(
            average_spectrum.logical_x_dimension,
            average_spectrum.logical_y_dimension,
            true,
        );
        ctf_values_image.allocate_2d(
            average_spectrum.logical_x_dimension,
            average_spectrum.logical_y_dimension,
            true,
        );
        spatial_frequency = vec![0.0; number_of_bins_in_1d_spectra as usize];
        rotational_average_astig = vec![0.0; number_of_bins_in_1d_spectra as usize];
        rotational_average_astig_renormalized = vec![0.0; number_of_bins_in_1d_spectra as usize];
        rotational_average_astig_fit = vec![0.0; number_of_bins_in_1d_spectra as usize];
        number_of_extrema_profile = vec![0.0; number_of_bins_in_1d_spectra as usize];
        ctf_values_profile = vec![0.0; number_of_bins_in_1d_spectra as usize];
        fit_frc = vec![0.0; number_of_bins_in_1d_spectra as usize];
        fit_frc_sigma = vec![0.0; number_of_bins_in_1d_spectra as usize];
        compute_images_with_number_of_extrema_and_ctf_values(
            &current_ctf,
            &mut number_of_extrema_image,
            &mut ctf_values_image,
        );
        // IMOD: return if error
        if !compute_rotational_average_of_power_spectrum(
            &average_spectrum,
            &current_ctf,
            &number_of_extrema_image,
            &ctf_values_image,
            number_of_bins_in_1d_spectra,
            &mut spatial_frequency,
            &mut rotational_average_astig,
            &mut rotational_average_astig_fit,
            &mut rotational_average_astig_renormalized,
            &mut number_of_extrema_profile,
            &mut ctf_values_profile,
        ) {
            return false;
        }

        // Here, do FRC
        let mut first_fit_bin = 0i32;
        for bin_counter in (0..number_of_bins_in_1d_spectra).rev() {
            if spatial_frequency[bin_counter as usize]
                >= f64::from(current_ctf.get_lowest_frequency_for_fitting())
            {
                first_fit_bin = bin_counter;
            }
        }
        compute_frc_between_1d_spectrum_and_fit(
            number_of_bins_in_1d_spectra,
            &rotational_average_astig_renormalized,
            &rotational_average_astig_fit,
            &number_of_extrema_profile,
            &mut fit_frc,
            &mut fit_frc_sigma,
            first_fit_bin,
        );

        // At what bin does CTF aliasing become problematic?
        last_bin_without_aliasing = 0;
        let mut location_of_previous_extremum = 0i32;
        for counter in 1..number_of_bins_in_1d_spectra {
            if number_of_extrema_profile[counter as usize]
                - number_of_extrema_profile[(counter - 1) as usize]
                >= 0.9
            {
                // We just reached a new extremum
                if counter - location_of_previous_extremum < 4 {
                    last_bin_without_aliasing = location_of_previous_extremum;
                    break;
                }
                location_of_previous_extremum = counter;
            }
        }
    }

    // Until what frequency were CTF rings detected?
    if compute_extra_stats {
        let low_threshold = 0.1f32;
        let frc_significance_threshold = 0.5f32;
        let high_threshold = 0.66f32;
        let mut at_last_bin_with_good_fit: bool;
        let mut number_of_bins_above_low_threshold = 0i32;
        let mut number_of_bins_above_significance_threshold = 0i32;
        let mut number_of_bins_above_high_threshold = 0i32;

        // Fix for IMOD: keep track of last unique value and fix tests at end
        let mut last_bin_with_unique_value = 0i32;
        let first_bin_to_check = (current_ctf
            .return_squared_spatial_frequency_of_a_zero(1, 0.0)
            .sqrt()
            * (average_spectrum.logical_x_dimension as f32))
            as i32;
        last_bin_with_good_fit = -1;

        // DNM: skip explicitly if there are no bins
        if first_bin_to_check >= number_of_bins_in_1d_spectra {
            last_bin_with_good_fit = 1;
        } else {
            for counter in first_bin_to_check..number_of_bins_in_1d_spectra {
                at_last_bin_with_good_fit = (number_of_bins_above_low_threshold > 3
                    && fit_frc[counter as usize] < f64::from(low_threshold))
                    || (number_of_bins_above_high_threshold > 3
                        && fit_frc[counter as usize] < f64::from(frc_significance_threshold));
                if at_last_bin_with_good_fit {
                    last_bin_with_good_fit = counter;
                    break;
                }
                // Count number of bins above given thresholds
                if fit_frc[counter as usize] > f64::from(low_threshold) {
                    number_of_bins_above_low_threshold += 1;
                }
                if fit_frc[counter as usize] > f64::from(frc_significance_threshold) {
                    number_of_bins_above_significance_threshold += 1;
                }
                if fit_frc[counter as usize] > f64::from(high_threshold) {
                    number_of_bins_above_high_threshold += 1;
                }
                if counter != 0
                    && (fit_frc[counter as usize] != fit_frc[(counter - 1) as usize]
                        || fit_frc[counter as usize] == 1.0)
                {
                    last_bin_with_unique_value = counter;
                }
            }
            if number_of_bins_above_significance_threshold
                == number_of_bins_in_1d_spectra - first_bin_to_check
                || (last_bin_with_good_fit < 0 && number_of_bins_above_high_threshold > 3)
            {
                last_bin_with_good_fit =
                    last_bin_with_unique_value.min(number_of_bins_in_1d_spectra - 1);
            }
            if number_of_bins_above_significance_threshold == 0 {
                last_bin_with_good_fit = 1;
            }
            last_bin_with_good_fit = last_bin_with_good_fit.min(number_of_bins_in_1d_spectra - 1);
        }
    } else {
        last_bin_with_good_fit = 1;
    }
    if last_bin_with_good_fit < 1 && last_bin_with_good_fit >= number_of_bins_in_1d_spectra {
        last_bin_with_good_fit = 1;
    }

    // Prepare output diagnostic image
    // DNM 3/31/23: do not call if no bins with good fit
    if compute_extra_stats
        && last_bin_with_good_fit > 1
        && !rescale_spectrum_and_rotational_average(
            &mut average_spectrum,
            &number_of_extrema_image,
            &ctf_values_image,
            number_of_bins_in_1d_spectra,
            &spatial_frequency,
            &mut rotational_average_astig,
            &rotational_average_astig_fit,
            &number_of_extrema_profile,
            &ctf_values_profile,
            last_bin_without_aliasing,
            last_bin_with_good_fit,
        )
    {
        return false;
    }

    // Send results back
    // Defocus 1 (Angstroms)
    results_array[0] = current_ctf.get_defocus_1() * pixel_size_for_fitting;
    // Defocus 2 (Angstroms)
    results_array[1] = current_ctf.get_defocus_2() * pixel_size_for_fitting;
    // Astigmatism angle (degrees)
    results_array[2] = (f64::from(current_ctf.get_astigmatism_azimuth()) * 180.0 / PI) as f32;
    // Additional phase shift (e.g. from phase plate) (radians)
    results_array[3] = current_ctf.get_additional_phase_shift();
    // CTFFIND score
    results_array[4] = -final_best_score;
    // IMOD: add || !compute_extra_stats to avoid uncomputed items
    if last_bin_with_good_fit == 0 || !compute_extra_stats {
        // A value of 0.0 indicates that the calculation to determine the
        // goodness of fit failed for some reason
        results_array[5] = 0.0;
    } else {
        // The resolution (Angstroms) up to which Thon rings are well fit
        results_array[5] = (f64::from(pixel_size_for_fitting)
            / spatial_frequency[last_bin_with_good_fit as usize]) as f32;
    }
    if last_bin_without_aliasing == 0 || !compute_extra_stats {
        // A value of 0.0 indicates that no aliasing was detected
        results_array[6] = 0.0;
    } else {
        // The resolution (Angstroms) at which aliasing was just detected
        results_array[6] = (f64::from(pixel_size_for_fitting)
            / spatial_frequency[last_bin_without_aliasing as usize])
            as f32;
    }

    // IMOD: Send back rotational average and other parameters
    if let Some(out) = rotational_avg_out.as_deref_mut() {
        let mut values = vec![0.0f32; number_of_bins_in_1d_spectra as usize];
        for counter in 0..number_of_bins_in_1d_spectra as usize {
            values[counter] = rotational_average.data_x[counter];
        }
        *out = Some(values);
    }
    if let Some(out) = normalized_avg_out.as_deref_mut() {
        *out = None;
    }
    if let Some(out) = fit_curve_out.as_deref_mut() {
        *out = None;
    }
    *num_points_out = number_of_bins_in_1d_spectra;
    *last_bin_freq_out =
        ((f64::from(number_of_bins_in_1d_spectra) - 1.) / f64::from(box_size)) as f32;

    // And use the astigmatism result instead if extra-stats, plus fit and normalized
    if compute_extra_stats && normalized_avg_out.is_some() && fit_curve_out.is_some() {
        if let Some(out) = rotational_avg_out.as_deref_mut() {
            if let Some(values) = out.as_mut() {
                for counter in 0..number_of_bins_in_1d_spectra as usize {
                    values[counter] = rotational_average_astig[counter] as f32;
                }
            }
        }
        if let Some(out) = normalized_avg_out.as_deref_mut() {
            let mut values = vec![0.0f32; number_of_bins_in_1d_spectra as usize];
            for counter in 0..number_of_bins_in_1d_spectra as usize {
                values[counter] = rotational_average_astig_renormalized[counter] as f32;
            }
            *out = Some(values);
        }
        if let Some(out) = fit_curve_out.as_deref_mut() {
            let mut values = vec![0.0f32; number_of_bins_in_1d_spectra as usize];
            for counter in 0..number_of_bins_in_1d_spectra as usize {
                values[counter] = rotational_average_astig_fit[counter] as f32;
            }
            *out = Some(values);
        }
    }
    let _ = fit_frc_sigma;

    // Return
    true
}

/// C++ `Renormalize1DSpectrumForFRC` (`ctffind.cpp:1119`).
///
/// Go from an experimental radial average with decaying Thon rings to a
/// function between 0.0 and 1.0 for every oscillation, by ranking the values
/// in each interval between a zero and an extremum of the CTF.
pub fn renormalize_1d_spectrum_for_frc(
    number_of_bins: i32,
    average: &mut [f64],
    fit: &[f64],
    number_of_extrema_profile: &[f32],
) {
    let mut bin_of_previous_extremum: i32;
    let mut bin_of_current_extremum: i32;
    let mut bin_of_zero: i32;
    let mut temp_vector: Vec<f32> = Vec::new();
    let mut temp_ranks: Vec<usize>;

    bin_of_previous_extremum = 0;
    bin_of_current_extremum = 0;
    for bin_counter in 1..number_of_bins {
        if number_of_extrema_profile[bin_counter as usize]
            - number_of_extrema_profile[(bin_counter - 1) as usize]
            >= 0.9
        {
            // We just passed an extremum, at bin_counter-1
            bin_of_current_extremum = bin_counter - 1;
            if bin_of_previous_extremum > 0 {
                if (bin_of_current_extremum - bin_of_previous_extremum >= 4 && false)
                    || (number_of_extrema_profile[bin_counter as usize] < 7.0)
                {
                    // Loop from the previous extremum to the one we just found
                    // (there is a zero in between, let's find it)
                    bin_of_zero = (bin_of_current_extremum - bin_of_previous_extremum) / 2
                        + bin_of_previous_extremum;
                    for i in bin_of_previous_extremum..bin_of_current_extremum {
                        if fit[i as usize] < fit[(i - 1) as usize]
                            && fit[i as usize] < fit[(i + 1) as usize]
                        {
                            bin_of_zero = i;
                        }
                    }

                    // Now we can rank before the zero (the downslope)
                    temp_vector.clear();
                    for i in bin_of_previous_extremum..=bin_of_zero {
                        temp_vector.push(average[i as usize] as f32);
                    }
                    temp_ranks = rank_sort(&temp_vector);
                    for i in bin_of_previous_extremum..=bin_of_zero {
                        average[i as usize] = f64::from(
                            (temp_ranks[(i - bin_of_previous_extremum) as usize] as f32)
                                / ((temp_vector.len() - 1) as f32),
                        );
                        average[i as usize] = (average[i as usize] * PI * 0.5).sin();
                    }

                    // Now we can rank after the zero (upslope)
                    temp_vector.clear();
                    for i in bin_of_zero + 1..bin_of_current_extremum {
                        temp_vector.push(average[i as usize] as f32);
                    }
                    temp_ranks = rank_sort(&temp_vector);
                    for i in bin_of_zero + 1..bin_of_current_extremum {
                        average[i as usize] = f64::from(
                            ((temp_ranks[(i - bin_of_zero - 1) as usize] + 1) as f32)
                                / ((temp_vector.len() + 1) as f32),
                        );
                        average[i as usize] = (average[i as usize] * PI * 0.5).sin();
                    }
                } else {
                    // A simpler way, without ranking, is just normalize between
                    // 0.0 and 1.0
                    let mut min_value = 1.0f32;
                    let mut max_value = 0.0f32;
                    for i in bin_of_previous_extremum..bin_of_current_extremum {
                        if average[i as usize] > f64::from(max_value) {
                            max_value = average[i as usize] as f32;
                        }
                        if average[i as usize] < f64::from(min_value) {
                            min_value = average[i as usize] as f32;
                        }
                    }
                    for i in bin_of_previous_extremum..bin_of_current_extremum {
                        average[i as usize] -= f64::from(min_value);
                        if max_value - min_value > 0.0001 {
                            average[i as usize] /= f64::from(max_value - min_value);
                        }
                    }
                }
            }
            bin_of_previous_extremum = bin_of_current_extremum;
        }
    }
}

/// C++ `ComputeFRCBetween1DSpectrumAndFit` (`ctffind.cpp:1215`).
#[allow(clippy::too_many_arguments)]
pub fn compute_frc_between_1d_spectrum_and_fit(
    number_of_bins: i32,
    average: &[f64],
    fit: &[f64],
    number_of_extrema_profile: &[f32],
    frc: &mut [f64],
    frc_sigma: &mut [f64],
    first_fit_bin: i32,
) {
    // IMOD icl 11 switch to allocation
    let mut half_window_width = vec![0i32; number_of_bins as usize];
    let mut bin_of_previous_extremum: i32;
    let mut first_bin: i32;
    let mut last_bin: i32;
    let mut spectrum_mean: f64;
    let mut fit_mean: f64;
    let mut spectrum_sigma: f64;
    let mut fit_sigma: f64;
    let mut cross_product: f64;
    let mut number_of_bins_in_window: f32;

    let minimum_window_half_width = number_of_bins / 40;

    // DNM 3/29/23: Initialize in case there are no extrema and extend to the
    // rest only if an extremum is found
    for i in 1..number_of_bins as usize {
        half_window_width[i] = minimum_window_half_width;
    }

    // First, work out the size of the window over which we'll compute the FRC value
    bin_of_previous_extremum = 0;
    for bin_counter in 1..number_of_bins {
        if number_of_extrema_profile[bin_counter as usize]
            != number_of_extrema_profile[(bin_counter - 1) as usize]
        {
            for i in bin_of_previous_extremum..bin_counter {
                half_window_width[i as usize] = minimum_window_half_width.max(
                    ((1.0 + 0.1 * f64::from(number_of_extrema_profile[bin_counter as usize]))
                        * f64::from((bin_counter - bin_of_previous_extremum + 1) as f32))
                        as i32,
                );
                half_window_width[i as usize] =
                    half_window_width[i as usize].min(number_of_bins / 2 - 1);
            }
            bin_of_previous_extremum = bin_counter;
        }
    }
    half_window_width[0] = half_window_width[1];
    if bin_of_previous_extremum > 0 {
        for bin_counter in bin_of_previous_extremum..number_of_bins {
            half_window_width[bin_counter as usize] =
                half_window_width[(bin_of_previous_extremum - 1) as usize];
        }
    }

    // Now compute the FRC for each bin
    for bin_counter in 0..number_of_bins {
        if bin_counter < first_fit_bin {
            frc[bin_counter as usize] = 1.0;
        } else {
            spectrum_mean = 0.0;
            fit_mean = 0.0;
            spectrum_sigma = 0.0;
            fit_sigma = 0.0;
            cross_product = 0.0;
            // Work out the boundaries
            first_bin = bin_counter - half_window_width[bin_counter as usize];
            last_bin = bin_counter + half_window_width[bin_counter as usize];
            if first_bin < first_fit_bin {
                first_bin = first_fit_bin;
                last_bin = first_bin + 2 * half_window_width[bin_counter as usize] + 1;
            }
            if last_bin >= number_of_bins {
                last_bin = number_of_bins - 1;
                first_bin = last_bin - 2 * half_window_width[bin_counter as usize] - 1;
            }
            // First pass
            for i in first_bin..=last_bin {
                spectrum_mean += average[i as usize];
                fit_mean += fit[i as usize];
            }
            number_of_bins_in_window = (2 * half_window_width[bin_counter as usize] + 1) as f32;
            spectrum_mean /= f64::from(number_of_bins_in_window);
            fit_mean /= f64::from(number_of_bins_in_window);
            // Second pass
            for i in first_bin..=last_bin {
                cross_product +=
                    (average[i as usize] - spectrum_mean) * (fit[i as usize] - fit_mean);
                spectrum_sigma += (average[i as usize] - spectrum_mean).powi(2);
                fit_sigma += (fit[i as usize] - fit_mean).powi(2);
            }
            if spectrum_sigma > 0.0 && fit_sigma > 0.0 {
                frc[bin_counter as usize] = cross_product
                    / f64::from(
                        ((spectrum_sigma / f64::from(number_of_bins_in_window)) as f32).sqrt()
                            * ((fit_sigma / f64::from(number_of_bins_in_window)) as f32).sqrt(),
                    )
                    / f64::from(number_of_bins_in_window);
            } else {
                frc[bin_counter as usize] = 0.0;
            }
            frc_sigma[bin_counter as usize] = 2.0 / f64::from(number_of_bins_in_window.sqrt());
        }
    }
}

/// C++ `OverlayCTF` (`ctffind.cpp:1409`).
///
/// The IMOD build keeps the function but no longer calls it: the diagnostic
/// image it decorates is not written by `ctffind()`.
pub fn overlay_ctf(spectrum: &mut Image, ctf: &Ctf) {
    let mut values_in_rings = EmpiricalDistribution::new();
    let mut values_in_fitting_range = EmpiricalDistribution::new();
    let mut address: i64;
    let (mut i_logi, mut i_logi_sq): (f32, f32);
    let (mut j_logi, mut j_logi_sq): (f32, f32);
    let mut current_spatial_frequency_squared: f32;
    let mut current_azimuth: f32;
    let lowest_freq = f64::from(ctf.get_lowest_frequency_for_fitting()).powi(2) as f32;
    let highest_freq = f64::from(ctf.get_highest_frequency_for_fitting()).powi(2) as f32;
    let mut current_ctf_value: f32;

    address = 0;
    for j in 0..spectrum.logical_y_dimension {
        j_logi = ((j - spectrum.physical_address_of_box_center_y) as f32)
            * spectrum.fourier_voxel_size_y;
        j_logi_sq = j_logi.powf(2.0);
        for i in 0..spectrum.logical_x_dimension {
            i_logi = ((i - spectrum.physical_address_of_box_center_x) as f32)
                * spectrum.fourier_voxel_size_x;
            i_logi_sq = i_logi.powf(2.0);

            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

            if current_spatial_frequency_squared > lowest_freq
                && current_spatial_frequency_squared <= highest_freq
            {
                current_azimuth = j_logi.atan2(i_logi);
                current_ctf_value = ctf
                    .evaluate(current_spatial_frequency_squared, current_azimuth)
                    .abs();
                if current_ctf_value > 0.5 {
                    values_in_rings.add_sample_value(spectrum.real_values[address as usize]);
                }
                values_in_fitting_range.add_sample_value(spectrum.real_values[address as usize]);
                if j < spectrum.physical_address_of_box_center_y
                    && i < spectrum.physical_address_of_box_center_x
                {
                    spectrum.real_values[address as usize] = current_ctf_value;
                }
            }
            if current_spatial_frequency_squared <= lowest_freq {
                spectrum.real_values[address as usize] = 0.0;
            }
            address += 1;
        }
        address += i64::from(spectrum.padding_jump_value);
    }
}

/// C++ `RescaleSpectrumAndRotationalAverage` (`ctffind.cpp:1411`).
///
/// Rescale the spectrum and its 1D rotational average so that the peaks and
/// troughs are at 0.0 and 1.0.  The spectrum half is commented out in the IMOD
/// source, which needs only the 1D average rescaled.
#[allow(clippy::too_many_arguments)]
pub fn rescale_spectrum_and_rotational_average(
    spectrum: &mut Image,
    _number_of_extrema: &Image,
    _ctf_values: &Image,
    number_of_bins: i32,
    spatial_frequency: &[f64],
    average: &mut [f64],
    average_fit: &[f64],
    _number_of_extrema_profile: &[f32],
    _ctf_values_profile: &[f32],
    last_bin_without_aliasing: i32,
    last_bin_with_good_fit: i32,
) -> bool {
    let spectrum_is_blank = spectrum.is_constant();
    // This peak will be used as a renormalization.
    let rescale_based_on_maximum_number = 2;
    let sg_width = 7;
    let sg_order = 2;
    let mut background = vec![0.0f32; number_of_bins as usize];
    let mut peak = vec![0.0f32; number_of_bins as usize];
    let mut at_a_maximum: bool;
    let mut at_a_minimum: bool;
    let mut maximum_at_previous_bin: bool;
    let mut minimum_at_previous_bin: bool;
    let mut location_of_previous_maximum: i32;
    let mut location_of_previous_minimum: i32;
    let mut current_maximum_number = 0i32;
    // DNM 12/8/23: Initialize this to stop crashes, it will now skip rescaling
    // if it is never set
    let mut normalisation_bin_number = 0i32;
    let actually_do_rescaling: bool;
    let last_bin_to_rescale: i32;

    let mut minima_curve = Curve::new();
    let mut maxima_curve = Curve::new();

    // Initialise arrays and variables
    for bin_counter in 0..number_of_bins as usize {
        background[bin_counter] = 0.0;
        peak[bin_counter] = 0.0;
    }
    location_of_previous_maximum = 0;
    location_of_previous_minimum = 0;
    current_maximum_number = 0;
    at_a_maximum = false;
    // Note, this may not be true if we have the perfect phase plate
    at_a_minimum = true;

    if !spectrum_is_blank {
        for bin_counter in 1..number_of_bins - 1 {
            // Remember where we were before - minimum, maximum or neither
            maximum_at_previous_bin = at_a_maximum;
            minimum_at_previous_bin = at_a_minimum;
            // Are we at a CTF min or max?
            at_a_minimum = average_fit[bin_counter as usize]
                <= average_fit[(bin_counter - 1) as usize]
                && average_fit[bin_counter as usize] <= average_fit[(bin_counter + 1) as usize];
            at_a_maximum = average_fit[bin_counter as usize]
                >= average_fit[(bin_counter - 1) as usize]
                && average_fit[bin_counter as usize] >= average_fit[(bin_counter + 1) as usize];
            // It could be that the CTF is constant in this region, in which case
            // we stay at a minimum if we were there
            if at_a_maximum && at_a_minimum {
                at_a_minimum = minimum_at_previous_bin;
                at_a_maximum = maximum_at_previous_bin;
            }
            // Fill in values for the background or peak by linear interpolation
            if at_a_minimum {
                for i in location_of_previous_minimum + 1..=bin_counter {
                    background[i as usize] = (average[location_of_previous_minimum as usize]
                        * f64::from((bin_counter - i) as f32)
                        / f64::from((bin_counter - location_of_previous_minimum) as f32)
                        + average[bin_counter as usize]
                            * f64::from((i - location_of_previous_minimum) as f32)
                            / f64::from((bin_counter - location_of_previous_minimum) as f32))
                        as f32;
                }
                location_of_previous_minimum = bin_counter;
                minima_curve.add_point(
                    spatial_frequency[bin_counter as usize] as f32,
                    average[bin_counter as usize] as f32,
                );
            }
            if at_a_maximum {
                if !maximum_at_previous_bin && average_fit[bin_counter as usize] > 0.7 {
                    current_maximum_number += 1;
                }
                for i in location_of_previous_maximum + 1..=bin_counter {
                    peak[i as usize] = (average[location_of_previous_maximum as usize]
                        * f64::from((bin_counter - i) as f32)
                        / f64::from((bin_counter - location_of_previous_maximum) as f32)
                        + average[bin_counter as usize]
                            * f64::from((i - location_of_previous_maximum) as f32)
                            / f64::from((bin_counter - location_of_previous_maximum) as f32))
                        as f32;
                    if current_maximum_number == rescale_based_on_maximum_number {
                        normalisation_bin_number = bin_counter;
                    }
                }
                location_of_previous_maximum = bin_counter;
                maxima_curve.add_point(
                    spatial_frequency[bin_counter as usize] as f32,
                    average[bin_counter as usize] as f32,
                );
            }
            if at_a_maximum && at_a_minimum {
                // MyPrintfRed
                wx_printf(ANSI_COLOR_RED);
                wx_printf("Rescale spectrum: Error. At a minimum and a maximum simultaneously.");
                wx_printf(ANSI_COLOR_RESET);
                // IMOD: do not abort
                return false;
            }
        }

        // Fit the minima and maximum curves using Savitzky-Golay smoothing
        if maxima_curve.number_of_points > sg_width {
            maxima_curve.fit_savitzky_golay_to_data(sg_width, sg_order);
        }
        if minima_curve.number_of_points > sg_width {
            minima_curve.fit_savitzky_golay_to_data(sg_width, sg_order);
        }

        // Replace the background and peak envelopes with the smooth min/max curves
        for bin_counter in 0..number_of_bins as usize {
            if minima_curve.number_of_points > sg_width {
                background[bin_counter] = minima_curve.return_savitzky_golay_interpolation_from_x(
                    spatial_frequency[bin_counter] as f32,
                );
            }
            if maxima_curve.number_of_points > sg_width {
                peak[bin_counter] = maxima_curve.return_savitzky_golay_interpolation_from_x(
                    spatial_frequency[bin_counter] as f32,
                );
            }
        }

        // Now that we have worked out a background and a peak envelope, let's do
        // the actual rescaling
        actually_do_rescaling = (peak[normalisation_bin_number as usize]
            - background[normalisation_bin_number as usize])
            > 0.0;
        let _ = actually_do_rescaling;
        if last_bin_without_aliasing != 0 {
            last_bin_to_rescale = last_bin_with_good_fit.min(last_bin_without_aliasing);
        } else {
            last_bin_to_rescale = last_bin_with_good_fit;
        }
        let _ = last_bin_to_rescale;

        // Rescale the 1D average
        if peak[normalisation_bin_number as usize] > background[normalisation_bin_number as usize] {
            for bin_counter in 0..number_of_bins as usize {
                average[bin_counter] = (average[bin_counter] - f64::from(background[bin_counter]))
                    / f64::from(
                        peak[normalisation_bin_number as usize]
                            - background[normalisation_bin_number as usize],
                    )
                    * 0.95;
                // We want peaks to reach at least 0.1
                if (peak[bin_counter] - background[bin_counter]) < 0.1
                    && (peak[bin_counter] - background[bin_counter]).abs() > 0.000001
                    && (bin_counter as i32) <= last_bin_without_aliasing
                {
                    average[bin_counter] = average[bin_counter]
                        / f64::from(peak[bin_counter] - background[bin_counter])
                        * f64::from(
                            peak[normalisation_bin_number as usize]
                                - background[normalisation_bin_number as usize],
                        )
                        * 0.1;
                }
            }
        }
    } // end of test of spectrum_is_blank

    true
}

/// C++ `ComputeRotationalAverageOfPowerSpectrum` (`ctffind.cpp:1617`).
#[allow(clippy::too_many_arguments)]
pub fn compute_rotational_average_of_power_spectrum(
    spectrum: &Image,
    ctf: &Ctf,
    number_of_extrema: &Image,
    ctf_values: &Image,
    number_of_bins: i32,
    spatial_frequency: &mut [f64],
    average: &mut [f64],
    average_fit: &mut [f64],
    average_rank: &mut [f64],
    number_of_extrema_profile: &mut [f32],
    ctf_values_profile: &mut [f32],
) -> bool {
    let spectrum_is_blank = spectrum.is_constant();
    let min_angular_distances_from_axes_radians = (10.0 / 180.0 * PI) as f32;
    let mut azimuth_of_mid_defocus: f32;
    let angular_distance_from_axes: f32;
    let mut current_spatial_frequency_squared: f32;
    // IMOD icl 11 switch to allocation
    let mut number_of_values = vec![0i32; number_of_bins as usize];
    let mut address: i64;
    let mut chosen_bin: i32;

    // Initialise the output arrays
    for counter in 0..number_of_bins as usize {
        average[counter] = 0.0;
        average_fit[counter] = 0.0;
        average_rank[counter] = 0.0;
        ctf_values_profile[counter] = 0.0;
        number_of_values[counter] = 0;
    }

    if !spectrum_is_blank {
        // For each bin of our 1D profile we compute the CTF.  We choose the
        // azimuth to be mid way between the two defoci of the astigmatic CTF
        azimuth_of_mid_defocus = (f64::from(ctf.get_astigmatism_azimuth()) + PI * 0.25) as f32;
        // We don't want the azimuth too close to the axes
        // IMOD icl 11 add (float)
        angular_distance_from_axes = azimuth_of_mid_defocus % ((PI as f32) * 0.5f32);
        if angular_distance_from_axes.abs() < min_angular_distances_from_axes_radians {
            if angular_distance_from_axes > 0.0 {
                azimuth_of_mid_defocus = min_angular_distances_from_axes_radians;
            } else {
                azimuth_of_mid_defocus = -min_angular_distances_from_axes_radians;
            }
        }
        if f64::from(angular_distance_from_axes.abs())
            > 0.5 * PI - f64::from(min_angular_distances_from_axes_radians)
        {
            if angular_distance_from_axes > 0.0 {
                azimuth_of_mid_defocus =
                    (PI * 0.5 - f64::from(min_angular_distances_from_axes_radians)) as f32;
            } else {
                azimuth_of_mid_defocus =
                    (-PI * 0.5 + f64::from(min_angular_distances_from_axes_radians)) as f32;
            }
        }
        // Now that we've chosen an azimuth, we can compute the CTF for each bin
        for counter in 0..number_of_bins as usize {
            current_spatial_frequency_squared =
                f64::from((counter as f32) * spectrum.fourier_voxel_size_y).powi(2) as f32;
            spatial_frequency[counter] = f64::from(current_spatial_frequency_squared.sqrt());
            ctf_values_profile[counter] =
                ctf.evaluate(current_spatial_frequency_squared, azimuth_of_mid_defocus);
            number_of_extrema_profile[counter] = ctf
                .return_number_of_extrema_before_squared_spatial_frequency(
                    current_spatial_frequency_squared,
                    azimuth_of_mid_defocus,
                ) as f32;
        }

        // Now we can loop over the spectrum again and decide to which bin to add
        // each component
        address = 0;
        for j in 0..spectrum.logical_y_dimension {
            for i in 0..spectrum.logical_x_dimension {
                chosen_bin = return_spectrum_bin_number(
                    number_of_bins,
                    number_of_extrema_profile,
                    number_of_extrema,
                    address,
                    ctf_values,
                    ctf_values_profile,
                );
                // IMOD: return failure
                if chosen_bin < 0 {
                    wx_printf_fmt(
                        "ReturnSpectrumBinNumber failed to find bin for i = %d  j = %d\n",
                        &[CArg::Int(i64::from(i)), CArg::Int(i64::from(j))],
                    );
                    return false;
                }
                average[chosen_bin as usize] += f64::from(spectrum.real_values[address as usize]);
                number_of_values[chosen_bin as usize] += 1;
                address += 1;
            }
            address += i64::from(spectrum.padding_jump_value);
        }

        // Do the actual averaging
        for counter in 0..number_of_bins as usize {
            if number_of_values[counter] > 0 {
                average[counter] /= f64::from(number_of_values[counter] as f32);
            } else {
                average[counter] = 0.0;
            }
            average_fit[counter] = f64::from(ctf_values_profile[counter].abs());
        }
    }

    // Compute the rank version of the rotational average
    for counter in 0..number_of_bins as usize {
        average_rank[counter] = average[counter];
    }
    renormalize_1d_spectrum_for_frc(
        number_of_bins,
        average_rank,
        average_fit,
        number_of_extrema_profile,
    );
    true
}

/// C++ `ReturnSpectrumBinNumber` (`ctffind.cpp:1740`).
pub fn return_spectrum_bin_number(
    number_of_bins: i32,
    number_of_extrema_profile: &[f32],
    number_of_extrema: &Image,
    address: i64,
    ctf_values: &Image,
    ctf_values_profile: &[f32],
) -> i32 {
    let mut diff_number_of_extrema: f32;
    let mut diff_number_of_extrema_previous: f32;
    let mut diff_number_of_extrema_next: f32;
    let mut ctf_diff_from_current_bin: f32;
    let mut ctf_diff_from_current_bin_old: f32;
    let mut chosen_bin: i32;

    // Let's find the bin which has the same number of preceding extrema and the
    // most similar ctf value
    ctf_diff_from_current_bin = f32::MAX;
    chosen_bin = -1;
    for current_bin in 0..number_of_bins {
        diff_number_of_extrema = (number_of_extrema.real_values[address as usize]
            - number_of_extrema_profile[current_bin as usize])
            .abs();
        if current_bin > 0 {
            diff_number_of_extrema_previous = (number_of_extrema.real_values[address as usize]
                - number_of_extrema_profile[(current_bin - 1) as usize])
                .abs();
        } else {
            diff_number_of_extrema_previous = f32::MAX;
        }
        if current_bin < number_of_bins - 1 {
            diff_number_of_extrema_next = (number_of_extrema.real_values[address as usize]
                - number_of_extrema_profile[(current_bin + 1) as usize])
                .abs();
        } else {
            diff_number_of_extrema_next = f32::MAX;
        }

        if number_of_extrema.real_values[address as usize]
            > number_of_extrema_profile[(number_of_bins - 1) as usize]
        {
            chosen_bin = number_of_bins - 1;
        } else if diff_number_of_extrema <= 0.01
            || (diff_number_of_extrema < diff_number_of_extrema_previous
                && diff_number_of_extrema <= diff_number_of_extrema_next
                && number_of_extrema_profile[(current_bin - 1).max(0) as usize]
                    != number_of_extrema_profile
                        [(current_bin + 1).min(number_of_bins - 1) as usize])
        {
            // We're nearly there
            // Let's look for the position for the nearest CTF value
            ctf_diff_from_current_bin_old = ctf_diff_from_current_bin;
            ctf_diff_from_current_bin = (ctf_values.real_values[address as usize]
                - ctf_values_profile[current_bin as usize])
                .abs();
            if ctf_diff_from_current_bin < ctf_diff_from_current_bin_old {
                chosen_bin = current_bin;
            }
        }
    }
    if chosen_bin == -1 {
        // MyPrintfRed
        wx_printf(ANSI_COLOR_RED);
        wx_printf("Could not find bin\n");
        wx_printf(ANSI_COLOR_RESET);
        // IMOD: do not abort
        -1
    } else {
        chosen_bin
    }
}

/// C++ `ComputeImagesWithNumberOfExtremaAndCTFValues` (`ctffind.cpp:1878`).
///
/// Compute an image where each pixel stores the number of preceding CTF
/// extrema — image "E" in Rohou & Grigorieff 2015 (Fig 3).
pub fn compute_images_with_number_of_extrema_and_ctf_values(
    ctf: &Ctf,
    number_of_extrema: &mut Image,
    ctf_values: &mut Image,
) {
    let (mut i_logi, mut i_logi_sq): (f32, f32);
    let (mut j_logi, mut j_logi_sq): (f32, f32);
    let mut current_spatial_frequency_squared: f32;
    let mut current_azimuth: f32;
    let mut address: i64;

    address = 0;
    for j in 0..number_of_extrema.logical_y_dimension {
        j_logi = ((j - number_of_extrema.physical_address_of_box_center_y) as f32)
            * number_of_extrema.fourier_voxel_size_y;
        j_logi_sq = f64::from(j_logi).powi(2) as f32;
        for i in 0..number_of_extrema.logical_x_dimension {
            i_logi = ((i - number_of_extrema.physical_address_of_box_center_x) as f32)
                * number_of_extrema.fourier_voxel_size_x;
            i_logi_sq = f64::from(i_logi).powi(2) as f32;
            // Where are we?
            current_spatial_frequency_squared = j_logi_sq + i_logi_sq;
            if current_spatial_frequency_squared > 0.0 {
                current_azimuth = j_logi.atan2(i_logi);
            } else {
                current_azimuth = 0.0;
            }
            ctf_values.real_values[address as usize] =
                ctf.evaluate(current_spatial_frequency_squared, current_azimuth);
            number_of_extrema.real_values[address as usize] = ctf
                .return_number_of_extrema_before_squared_spatial_frequency(
                    current_spatial_frequency_squared,
                    current_azimuth,
                ) as f32;
            address += 1;
        }
        address += i64::from(number_of_extrema.padding_jump_value);
    }

    number_of_extrema.is_in_real_space = true;
    ctf_values.is_in_real_space = true;
}

/// C++ `FindRotationalAlignmentBetweenTwoStacksOfImages` (`ctffind.cpp:1927`).
///
/// Align rotationally a (stack) of image(s) against another image.  Return the
/// rotation angle that gives the best normalised cross-correlation.
pub fn find_rotational_alignment_between_two_stacks_of_images(
    self_image: &Image,
    other_image: &Image,
    number_of_images: i32,
    search_half_range: f32,
    search_step_size: f32,
    minimum_radius: f32,
    maximum_radius: f32,
) -> f32 {
    // Local variables
    let minimum_radius_sq = f64::from(minimum_radius).powi(2) as f32;
    let maximum_radius_sq = f64::from(maximum_radius).powi(2) as f32;
    let inverse_logical_x_dimension =
        (1.0 / f64::from(self_image.logical_x_dimension as f32)) as f32;
    let inverse_logical_y_dimension =
        (1.0 / f64::from(self_image.logical_y_dimension as f32)) as f32;
    let mut best_cc = -f32::MAX;
    let mut best_rotation = -f32::MAX;
    let mut current_rotation = -search_half_range;
    let mut current_rotation_rad: f32;
    let mut cc_numerator_dist = EmpiricalDistribution::new();
    let mut cc_denom_self_dist = EmpiricalDistribution::new();
    let mut cc_denom_other_dist = EmpiricalDistribution::new();
    let (mut i_logi, mut j_logi): (i32, i32);
    let (mut i_logi_frac, mut ii_phys): (f32, f32);
    let (mut j_logi_frac, mut jj_phys): (f32, f32);
    let mut current_interpolated_value: f32;
    let mut address_in_other_image: i64;
    let mut current_cc: f32;

    // Loop over possible rotations
    while current_rotation < search_half_range + search_step_size {
        current_rotation_rad = (f64::from(current_rotation) / 180.0 * PI) as f32;
        cc_numerator_dist.reset();
        cc_denom_self_dist.reset();
        cc_denom_other_dist.reset();
        // Loop over the array of images
        for _current_image in 0..number_of_images {
            // Loop over the other (reference) image
            address_in_other_image = 0;
            for j in 0..other_image.logical_y_dimension {
                j_logi = j - other_image.physical_address_of_box_center_y;
                j_logi_frac =
                    f64::from((j_logi as f32) * inverse_logical_y_dimension).powi(2) as f32;
                for i in 0..other_image.logical_x_dimension {
                    i_logi = i - other_image.physical_address_of_box_center_x;
                    i_logi_frac = (f64::from((i_logi as f32) * inverse_logical_x_dimension).powi(2)
                        + f64::from(j_logi_frac)) as f32;

                    if i_logi_frac >= minimum_radius_sq && i_logi_frac <= maximum_radius_sq {
                        // We do ccw rotation to go from other_image (reference)
                        // to self (input image)
                        ii_phys = (i_logi as f32) * current_rotation_rad.cos()
                            - (j_logi as f32) * current_rotation_rad.sin()
                            + self_image.physical_address_of_box_center_x as f32;
                        jj_phys = (i_logi as f32) * current_rotation_rad.sin()
                            + (j_logi as f32) * current_rotation_rad.cos()
                            + self_image.physical_address_of_box_center_y as f32;

                        if (ii_phys as i32) > 0
                            && (ii_phys as i32) + 1 < self_image.logical_x_dimension
                            && (jj_phys as i32) > 0
                            && (jj_phys as i32) + 1 < self_image.logical_y_dimension
                        {
                            current_interpolated_value = 0.0;
                            self_image
                                .get_real_value_by_linear_interpolation_no_bounds_check_image(
                                    ii_phys,
                                    jj_phys,
                                    &mut current_interpolated_value,
                                );
                            cc_numerator_dist.add_sample_value(
                                current_interpolated_value
                                    * other_image.real_values[address_in_other_image as usize],
                            );
                            cc_denom_other_dist.add_sample_value(
                                f64::from(other_image.real_values[address_in_other_image as usize])
                                    .powi(2) as f32,
                            );
                            cc_denom_self_dist.add_sample_value(
                                f64::from(current_interpolated_value).powi(2) as f32,
                            );
                        }
                    }
                    address_in_other_image += 1;
                } // i
                address_in_other_image += i64::from(other_image.padding_jump_value);
            } // end of loop over other (reference) image
        } // end of loop over array of images

        current_cc = cc_numerator_dist.get_sample_sum()
            / (cc_denom_other_dist.get_sample_sum() * cc_denom_self_dist.get_sample_sum()).sqrt();

        if current_cc > best_cc {
            best_cc = current_cc;
            best_rotation = current_rotation;
        }

        // Increment the rotation
        current_rotation += search_step_size;
    } // end of loop over rotations

    best_rotation
}
