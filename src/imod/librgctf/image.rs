//! Translation of `IMOD/librgctf/image.h` and `IMOD/librgctf/image_trim.cpp`.
//!
//! There is no `image.cpp` in this revision: `image_trim.cpp` is cisTEM's
//! `image.cpp` with the sections CTFFIND does not need commented out, and the
//! commented-out members are not translated here either — including
//! `ForwardFFT` and `BackwardFFT`, whose bodies are entirely inside the
//! `/* IMOD ... */` block, so in this library they do nothing at all.
//!
//! `real_values` and `complex_values` are two views of **one** `malloc`
//! allocation in the C++ (`complex_values = (std::complex<float>*)
//! real_values`).  That aliasing is behaviour — `CopyFrom` moves Fourier data
//! by copying `real_memory_allocated` floats — so the Rust image keeps a
//! single `Vec<f32>` and reads or writes complex element `k` as the float pair
//! at `2k` and `2k + 1`.

use super::ctf::Ctf;
use super::curve::Curve;
use super::defines::PI;
use super::empirical_distribution::EmpiricalDistribution;
use super::functions::is_even;
use super::globals::global_random_number_generator;
use super::types::{Complex, I};
use std::sync::Mutex;

/// C++ `WriteSliceType` (`functions.h:34`): `int (*)(const char *, float *, int, int)`.
pub type WriteSliceType = fn(&str, &[f32], i32, i32) -> i32;

/// C++ `static WriteSliceType sSliceWriteFunc = NULL` (`image_trim.cpp:1281`).
static S_SLICE_WRITE_FUNC: Mutex<Option<WriteSliceType>> = Mutex::new(None);

/// C++ `internalSetWriteSliceFunc` (`image_trim.cpp:1283`).
pub fn internal_set_write_slice_func(func: Option<WriteSliceType>) {
    *S_SLICE_WRITE_FUNC
        .lock()
        .expect("slice write callback lock poisoned") = func;
}

/// C++ `Image` (`image.h:12`).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub logical_x_dimension: i32,
    pub logical_y_dimension: i32,
    pub logical_z_dimension: i32,

    pub is_in_real_space: bool,
    pub object_is_centred_in_box: bool,

    pub physical_upper_bound_complex_x: i32,
    pub physical_upper_bound_complex_y: i32,
    pub physical_upper_bound_complex_z: i32,

    pub physical_address_of_box_center_x: i32,
    pub physical_address_of_box_center_y: i32,
    pub physical_address_of_box_center_z: i32,

    pub physical_index_of_first_negative_frequency_y: i32,
    pub physical_index_of_first_negative_frequency_z: i32,

    pub fourier_voxel_size_x: f32,
    pub fourier_voxel_size_y: f32,
    pub fourier_voxel_size_z: f32,

    pub logical_upper_bound_complex_x: i32,
    pub logical_upper_bound_complex_y: i32,
    pub logical_upper_bound_complex_z: i32,

    pub logical_lower_bound_complex_x: i32,
    pub logical_lower_bound_complex_y: i32,
    pub logical_lower_bound_complex_z: i32,

    pub logical_upper_bound_real_x: i32,
    pub logical_upper_bound_real_y: i32,
    pub logical_upper_bound_real_z: i32,

    pub logical_lower_bound_real_x: i32,
    pub logical_lower_bound_real_y: i32,
    pub logical_lower_bound_real_z: i32,

    pub real_memory_allocated: i64,

    pub padding_jump_value: i32,

    pub insert_into_which_reconstruction: i32,

    pub number_of_real_space_pixels: i64,
    pub ft_normalization_factor: f32,

    /// The single allocation behind both `real_values` and `complex_values`.
    pub real_values: Vec<f32>,
    pub is_in_memory: bool,

    pub image_memory_should_not_be_deallocated: bool,
}

impl Default for Image {
    fn default() -> Self {
        Self::new()
    }
}

impl Image {
    /// C++ `Image::Image()` (`image_trim.cpp:65`).
    pub fn new() -> Self {
        let mut this = Self {
            logical_x_dimension: 0,
            logical_y_dimension: 0,
            logical_z_dimension: 0,
            is_in_real_space: true,
            object_is_centred_in_box: true,
            physical_upper_bound_complex_x: 0,
            physical_upper_bound_complex_y: 0,
            physical_upper_bound_complex_z: 0,
            physical_address_of_box_center_x: 0,
            physical_address_of_box_center_y: 0,
            physical_address_of_box_center_z: 0,
            physical_index_of_first_negative_frequency_y: 0,
            physical_index_of_first_negative_frequency_z: 0,
            fourier_voxel_size_x: 0.0,
            fourier_voxel_size_y: 0.0,
            fourier_voxel_size_z: 0.0,
            logical_upper_bound_complex_x: 0,
            logical_upper_bound_complex_y: 0,
            logical_upper_bound_complex_z: 0,
            logical_lower_bound_complex_x: 0,
            logical_lower_bound_complex_y: 0,
            logical_lower_bound_complex_z: 0,
            logical_upper_bound_real_x: 0,
            logical_upper_bound_real_y: 0,
            logical_upper_bound_real_z: 0,
            logical_lower_bound_real_x: 0,
            logical_lower_bound_real_y: 0,
            logical_lower_bound_real_z: 0,
            real_memory_allocated: 0,
            padding_jump_value: 0,
            insert_into_which_reconstruction: 0,
            number_of_real_space_pixels: 0,
            ft_normalization_factor: 0.0,
            real_values: Vec::new(),
            is_in_memory: false,
            image_memory_should_not_be_deallocated: false,
        };
        this.setup_initial_values();
        this
    }

    /// C++ `Image::SetupInitialValues` (`image_trim.cpp:7`).
    pub fn setup_initial_values(&mut self) {
        self.logical_x_dimension = 0;
        self.logical_y_dimension = 0;
        self.logical_z_dimension = 0;

        self.is_in_real_space = true;
        self.object_is_centred_in_box = true;

        self.physical_upper_bound_complex_x = 0;
        self.physical_upper_bound_complex_y = 0;
        self.physical_upper_bound_complex_z = 0;

        self.physical_address_of_box_center_x = 0;
        self.physical_address_of_box_center_y = 0;
        self.physical_address_of_box_center_z = 0;

        self.physical_index_of_first_negative_frequency_y = 0;
        self.physical_index_of_first_negative_frequency_z = 0;

        self.fourier_voxel_size_x = 0.0;
        self.fourier_voxel_size_y = 0.0;
        self.fourier_voxel_size_z = 0.0;

        self.logical_upper_bound_complex_x = 0;
        self.logical_upper_bound_complex_y = 0;
        self.logical_upper_bound_complex_z = 0;

        self.logical_lower_bound_complex_x = 0;
        self.logical_lower_bound_complex_y = 0;
        self.logical_lower_bound_complex_z = 0;

        self.logical_upper_bound_real_x = 0;
        self.logical_upper_bound_real_y = 0;
        self.logical_upper_bound_real_z = 0;

        self.logical_lower_bound_real_x = 0;
        self.logical_lower_bound_real_y = 0;
        self.logical_lower_bound_real_z = 0;

        self.insert_into_which_reconstruction = 0;
        self.real_values = Vec::new();

        self.is_in_memory = false;
        self.real_memory_allocated = 0;

        self.padding_jump_value = 0;
        self.image_memory_should_not_be_deallocated = false;
    }

    /// The complex view of element `index` of the shared allocation.
    #[inline]
    pub fn complex_values(&self, index: i64) -> Complex {
        let base = (index * 2) as usize;
        Complex::new(self.real_values[base], self.real_values[base + 1])
    }

    /// Stores into the complex view of element `index`.
    #[inline]
    pub fn set_complex_values(&mut self, index: i64, value: Complex) {
        let base = (index * 2) as usize;
        self.real_values[base] = value.re;
        self.real_values[base + 1] = value.im;
    }

    /// C++ `Image::ReturnSmallestLogicalDimension` (`image_trim.cpp:84`).
    pub fn return_smallest_logical_dimension(&self) -> i32 {
        if self.logical_z_dimension == 1 {
            self.logical_x_dimension.min(self.logical_y_dimension)
        } else {
            let temp_int = self.logical_x_dimension.min(self.logical_y_dimension);
            temp_int.min(self.logical_z_dimension)
        }
    }

    /// C++ `Image::ReturnLargestLogicalDimension` (`image_trim.cpp:99`).
    pub fn return_largest_logical_dimension(&self) -> i32 {
        if self.logical_z_dimension == 1 {
            self.logical_x_dimension.max(self.logical_y_dimension)
        } else {
            let temp_int = self.logical_x_dimension.max(self.logical_y_dimension);
            temp_int.max(self.logical_z_dimension)
        }
    }

    /// C++ `Image::MultiplyPixelWise` (`image_trim.cpp:113`).
    pub fn multiply_pixel_wise(&mut self, other_image: &Image) {
        if self.is_in_real_space {
            for pixel_counter in 0..self.real_memory_allocated as usize {
                self.real_values[pixel_counter] *= other_image.real_values[pixel_counter];
            }
        } else {
            for pixel_counter in 0..(self.real_memory_allocated / 2) {
                let value =
                    self.complex_values(pixel_counter) * other_image.complex_values(pixel_counter);
                self.set_complex_values(pixel_counter, value);
            }
        }
    }

    /// C++ `Image::CircleMask` (`image_trim.cpp:141`).
    pub fn circle_mask(&mut self, wanted_mask_radius: f32, invert: bool) {
        let mut pixel_counter: i64;
        let (mut x, mut y, mut z): (f32, f32, f32);
        let mut distance_from_center_squared: f32;
        let wanted_mask_radius_squared = wanted_mask_radius.powf(2.0);
        let mut average_value = 0.0f64;
        let mut number_of_pixels = 0i64;

        pixel_counter = 0;
        for k in 0..self.logical_z_dimension {
            z = ((k - self.physical_address_of_box_center_z) as f32).powf(2.0);

            for j in 0..self.logical_y_dimension {
                y = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);

                for i in 0..self.logical_x_dimension {
                    x = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared - wanted_mask_radius_squared).abs() <= 4.0 {
                        number_of_pixels += 1;
                        average_value += f64::from(self.real_values[pixel_counter as usize]);
                    }

                    pixel_counter += 1;
                }
                pixel_counter += i64::from(self.padding_jump_value);
            }
        }

        // Now we know what value to mask with
        average_value /= f64::from(number_of_pixels as f32);

        // Let's mask
        pixel_counter = 0;
        for k in 0..self.logical_z_dimension {
            z = ((k - self.physical_address_of_box_center_z) as f32).powf(2.0);

            for j in 0..self.logical_y_dimension {
                y = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);

                for i in 0..self.logical_x_dimension {
                    x = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);

                    distance_from_center_squared = x + y + z;

                    if invert {
                        if distance_from_center_squared <= wanted_mask_radius_squared {
                            self.real_values[pixel_counter as usize] = average_value as f32;
                        }
                    } else if distance_from_center_squared > wanted_mask_radius_squared {
                        self.real_values[pixel_counter as usize] = average_value as f32;
                    }

                    pixel_counter += 1;
                }
                pixel_counter += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::CosineMask` (`image_trim.cpp:227`).
    pub fn cosine_mask(
        &mut self,
        wanted_mask_radius: f32,
        wanted_mask_edge: f32,
        invert: bool,
        force_mask_value: bool,
        wanted_mask_value: f32,
    ) -> f32 {
        let (mut ii, mut jj, mut kk): (i32, i32, i32);
        let mut number_of_pixels: i64;

        let (mut x, mut y, mut z): (f32, f32, f32);

        let mut pixel_counter: i64 = 0;

        let mut distance_from_center: f32;
        let mask_radius_plus_edge: f32;
        let mut distance_from_center_squared: f32;
        let mut mask_radius: f32;
        let mask_radius_squared: f32;
        let mask_radius_plus_edge_squared: f32;
        let mut edge: f32;
        let mut pixel_sum: f64;

        let mut frequency: f32;
        let mut frequency_squared: f32;

        let mut mask_volume = 0.0f64;

        mask_radius = wanted_mask_radius - wanted_mask_edge / 2.0;
        if mask_radius < 0.0 {
            mask_radius = 0.0;
        }
        mask_radius_plus_edge = mask_radius + wanted_mask_edge;

        mask_radius_squared = mask_radius.powf(2.0);
        mask_radius_plus_edge_squared = mask_radius_plus_edge.powf(2.0);

        pixel_sum = 0.0;
        number_of_pixels = 0;
        if self.is_in_real_space && self.object_is_centred_in_box {
            if force_mask_value {
                pixel_sum = f64::from(wanted_mask_value);
            } else {
                for k in 0..self.logical_z_dimension {
                    z = ((k - self.physical_address_of_box_center_z) as f32).powf(2.0);

                    for j in 0..self.logical_y_dimension {
                        y = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);

                        for i in 0..self.logical_x_dimension {
                            x = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);

                            distance_from_center_squared = x + y + z;

                            if distance_from_center_squared >= mask_radius_squared
                                && distance_from_center_squared <= mask_radius_plus_edge_squared
                            {
                                pixel_sum += f64::from(self.real_values[pixel_counter as usize]);
                                number_of_pixels += 1;
                            }
                            pixel_counter += 1;
                        }
                        pixel_counter += i64::from(self.padding_jump_value);
                    }
                }
                pixel_sum /= number_of_pixels as f64;
            }

            pixel_counter = 0;
            for k in 0..self.logical_z_dimension {
                z = ((k - self.physical_address_of_box_center_z) as f32).powf(2.0);

                for j in 0..self.logical_y_dimension {
                    y = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);

                    for i in 0..self.logical_x_dimension {
                        x = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);

                        distance_from_center_squared = x + y + z;

                        if distance_from_center_squared >= mask_radius_squared
                            && distance_from_center_squared <= mask_radius_plus_edge_squared
                        {
                            distance_from_center = distance_from_center_squared.sqrt();
                            edge = ((1.0
                                + f64::from(
                                    ((PI * f64::from(distance_from_center - mask_radius)
                                        / f64::from(wanted_mask_edge))
                                        as f32)
                                        .cos(),
                                ))
                                / 2.0) as f32;
                            if invert {
                                self.real_values[pixel_counter as usize] =
                                    (f64::from(self.real_values[pixel_counter as usize])
                                        * (1.0 - f64::from(edge))
                                        + f64::from(edge) * pixel_sum)
                                        as f32;
                                mask_volume +=
                                    f64::from(((1.0 - f64::from(edge)) as f32).powf(2.0));
                            } else {
                                self.real_values[pixel_counter as usize] =
                                    (f64::from(self.real_values[pixel_counter as usize])
                                        * f64::from(edge)
                                        + (1.0 - f64::from(edge)) * pixel_sum)
                                        as f32;
                                mask_volume += f64::from(edge.powf(2.0));
                            }
                        } else if invert {
                            if distance_from_center_squared <= mask_radius_squared {
                                self.real_values[pixel_counter as usize] = pixel_sum as f32;
                            } else {
                                mask_volume += 1.0;
                            }
                        } else if distance_from_center_squared >= mask_radius_plus_edge_squared {
                            self.real_values[pixel_counter as usize] = pixel_sum as f32;
                        } else {
                            mask_volume += 1.0;
                        }

                        pixel_counter += 1;
                    }
                    pixel_counter += i64::from(self.padding_jump_value);
                }
            }
        } else if self.is_in_real_space {
            if force_mask_value {
                pixel_sum = f64::from(wanted_mask_value);
            } else {
                for k in 0..self.logical_z_dimension {
                    kk = k;
                    if kk >= self.physical_address_of_box_center_z {
                        kk -= self.logical_z_dimension;
                    }
                    z = (kk as f32).powf(2.0);

                    for j in 0..self.logical_y_dimension {
                        jj = j;
                        if jj >= self.physical_address_of_box_center_y {
                            jj -= self.logical_y_dimension;
                        }
                        y = (jj as f32).powf(2.0);

                        for i in 0..self.logical_x_dimension {
                            ii = i;
                            if ii >= self.physical_address_of_box_center_x {
                                ii -= self.logical_x_dimension;
                            }
                            x = (ii as f32).powf(2.0);

                            distance_from_center_squared = x + y + z;

                            if distance_from_center_squared >= mask_radius_squared
                                && distance_from_center_squared <= mask_radius_plus_edge_squared
                            {
                                pixel_sum += f64::from(self.real_values[pixel_counter as usize]);
                                number_of_pixels += 1;
                            }
                            pixel_counter += 1;
                        }
                        pixel_counter += i64::from(self.padding_jump_value);
                    }
                }
                pixel_sum /= number_of_pixels as f64;
            }

            pixel_counter = 0;
            for k in 0..self.logical_z_dimension {
                kk = k;
                if kk >= self.physical_address_of_box_center_z {
                    kk -= self.logical_z_dimension;
                }
                z = (kk as f32).powf(2.0);

                for j in 0..self.logical_y_dimension {
                    jj = j;
                    if jj >= self.physical_address_of_box_center_y {
                        jj -= self.logical_y_dimension;
                    }
                    y = (jj as f32).powf(2.0);

                    for i in 0..self.logical_x_dimension {
                        ii = i;
                        if ii >= self.physical_address_of_box_center_x {
                            ii -= self.logical_x_dimension;
                        }
                        x = (ii as f32).powf(2.0);

                        distance_from_center_squared = x + y + z;

                        if distance_from_center_squared >= mask_radius_squared
                            && distance_from_center_squared <= mask_radius_plus_edge_squared
                        {
                            distance_from_center = distance_from_center_squared.sqrt();
                            edge = ((1.0
                                + f64::from(
                                    ((PI * f64::from(distance_from_center - mask_radius)
                                        / f64::from(wanted_mask_edge))
                                        as f32)
                                        .cos(),
                                ))
                                / 2.0) as f32;
                            self.real_values[pixel_counter as usize] =
                                (f64::from(self.real_values[pixel_counter as usize])
                                    * f64::from(edge)
                                    + (1.0 - f64::from(edge)) * pixel_sum)
                                    as f32;
                            mask_volume += f64::from(edge.powf(2.0));
                        } else if distance_from_center_squared >= mask_radius_plus_edge_squared {
                            self.real_values[pixel_counter as usize] = pixel_sum as f32;
                        } else {
                            mask_volume += 1.0;
                        }

                        pixel_counter += 1;
                    }
                    pixel_counter += i64::from(self.padding_jump_value);
                }
            }
        } else {
            for k in 0..=self.physical_upper_bound_complex_z {
                z = ((self.return_fourier_logical_coord_given_physical_coord_z(k) as f32
                    * self.fourier_voxel_size_z) as f32)
                    .powf(2.0);

                for j in 0..=self.physical_upper_bound_complex_y {
                    y = (self.return_fourier_logical_coord_given_physical_coord_y(j) as f32
                        * self.fourier_voxel_size_y)
                        .powf(2.0);

                    for i in 0..=self.physical_upper_bound_complex_x {
                        x = (i as f32 * self.fourier_voxel_size_x).powf(2.0);

                        // compute squared radius, in units of reciprocal pixels

                        frequency_squared = x + y + z;

                        if frequency_squared >= mask_radius_squared
                            && frequency_squared <= mask_radius_plus_edge_squared
                        {
                            frequency = frequency_squared.sqrt();
                            edge = ((1.0
                                + f64::from(
                                    ((PI * f64::from(frequency - mask_radius)
                                        / f64::from(wanted_mask_edge))
                                        as f32)
                                        .cos(),
                                ))
                                / 2.0) as f32;
                            let value = self.complex_values(pixel_counter);
                            if invert {
                                self.set_complex_values(
                                    pixel_counter,
                                    value * ((1.0 - f64::from(edge)) as f32),
                                );
                            } else {
                                self.set_complex_values(pixel_counter, value * edge);
                            }
                        }
                        if invert {
                            if frequency_squared <= mask_radius_squared {
                                self.set_complex_values(
                                    pixel_counter,
                                    Complex::new(0.0, 0.0) + I * 0.0f32,
                                );
                            }
                        } else if frequency_squared >= mask_radius_plus_edge_squared {
                            self.set_complex_values(
                                pixel_counter,
                                Complex::new(0.0, 0.0) + I * 0.0f32,
                            );
                        }

                        pixel_counter += 1;
                    }
                }
            }
        }

        mask_volume as f32
    }

    /// C++ `Image::operator = (const Image *)` (`image_trim.cpp:490`).
    pub fn assign(&mut self, other_image: &Image) {
        // Check for self assignment
        if std::ptr::eq(self, other_image) {
            return;
        }

        if self.is_in_memory {
            if self.logical_x_dimension != other_image.logical_x_dimension
                || self.logical_y_dimension != other_image.logical_y_dimension
                || self.logical_z_dimension != other_image.logical_z_dimension
            {
                self.deallocate();
                self.allocate(
                    other_image.logical_x_dimension,
                    other_image.logical_y_dimension,
                    other_image.logical_z_dimension,
                    other_image.is_in_real_space,
                );
            }
        } else {
            self.allocate(
                other_image.logical_x_dimension,
                other_image.logical_y_dimension,
                other_image.logical_z_dimension,
                other_image.is_in_real_space,
            );
        }

        // by here the memory allocation should be ok..

        self.is_in_real_space = other_image.is_in_real_space;
        self.object_is_centred_in_box = other_image.object_is_centred_in_box;

        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] = other_image.real_values[pixel_counter];
        }
    }

    /// C++ `Image::Deallocate` (`image_trim.cpp:546`).
    pub fn deallocate(&mut self) {
        if self.is_in_memory && !self.image_memory_should_not_be_deallocated {
            self.real_values = Vec::new();
            self.is_in_memory = false;
        }
    }

    /// C++ `Image::Allocate(int, int, int, bool)` (`image_trim.cpp:572`).
    pub fn allocate(
        &mut self,
        wanted_x_size: i32,
        wanted_y_size: i32,
        wanted_z_size: i32,
        should_be_in_real_space: bool,
    ) {
        // check to see if we need to do anything?

        if self.is_in_memory {
            self.is_in_real_space = should_be_in_real_space;

            if wanted_x_size == self.logical_x_dimension
                && wanted_y_size == self.logical_y_dimension
                && wanted_z_size == self.logical_z_dimension
            {
                // everything is already done..
                self.is_in_real_space = should_be_in_real_space;
                return;
            } else {
                self.deallocate();
            }
        }

        // if we got here we need to do allocation..

        self.set_logical_dimensions(wanted_x_size, wanted_y_size, wanted_z_size);
        self.is_in_real_space = should_be_in_real_space;

        // first_x_dimension
        if is_even(wanted_x_size) {
            self.real_memory_allocated = i64::from(wanted_x_size / 2 + 1);
        } else {
            self.real_memory_allocated = i64::from((wanted_x_size - 1) / 2 + 1);
        }

        self.real_memory_allocated *= i64::from(wanted_y_size) * i64::from(wanted_z_size);
        self.real_memory_allocated *= 2; // room for complex

        // The C++ uses `malloc`, which leaves the block uninitialised; a `Vec`
        // is zeroed.  Every caller in this library writes before it reads.
        self.real_values = vec![0.0; self.real_memory_allocated as usize];

        self.is_in_memory = true;

        // Update addresses etc..
        self.update_looping_and_addressing();

        // set the loop junk value..
        if is_even(self.logical_x_dimension) {
            self.padding_jump_value = 2;
        } else {
            self.padding_jump_value = 1;
        }

        self.number_of_real_space_pixels = i64::from(self.logical_x_dimension)
            * i64::from(self.logical_y_dimension)
            * i64::from(self.logical_z_dimension);
        self.ft_normalization_factor = 1.0 / (self.number_of_real_space_pixels as f32).sqrt();
    }

    /// C++ `Image::Allocate(int, int, bool)` (`image_trim.cpp:656`).
    pub fn allocate_2d(
        &mut self,
        wanted_x_size: i32,
        wanted_y_size: i32,
        should_be_in_real_space: bool,
    ) {
        self.allocate(wanted_x_size, wanted_y_size, 1, should_be_in_real_space);
    }

    /// C++ `Image::AllocateAsPointingToSliceIn3D` (`image_trim.cpp:662`).
    ///
    /// The C++ makes `real_values` point *into* the 3D image's allocation, so
    /// writes travel both ways.  Rust cannot express that with an owned `Vec`,
    /// so this copies the slice and leaves
    /// `image_memory_should_not_be_deallocated` set as the source does.  The
    /// function is dead in the C too — nothing in `librgctf` calls it — and the
    /// deviation is recorded here rather than hidden.
    pub fn allocate_as_pointing_to_slice_in_3d(&mut self, wanted3d: &Image, wanted_slice: i64) {
        self.deallocate();
        self.is_in_real_space = wanted3d.is_in_real_space;

        self.set_logical_dimensions(
            wanted3d.logical_x_dimension,
            wanted3d.logical_y_dimension,
            1,
        );

        let bytes_in_slice =
            wanted3d.real_memory_allocated / i64::from(wanted3d.logical_z_dimension);

        self.image_memory_should_not_be_deallocated = true;
        self.is_in_memory = true; // kind of a lie
        self.real_memory_allocated = bytes_in_slice; // kind of a lie

        let start = (bytes_in_slice * (wanted_slice - 1)) as usize;
        self.real_values = wanted3d.real_values[start..start + bytes_in_slice as usize].to_vec();

        self.update_looping_and_addressing();

        if is_even(self.logical_x_dimension) {
            self.padding_jump_value = 2;
        } else {
            self.padding_jump_value = 1;
        }

        self.number_of_real_space_pixels = i64::from(self.logical_x_dimension)
            * i64::from(self.logical_y_dimension)
            * i64::from(self.logical_z_dimension);
        self.ft_normalization_factor = 1.0 / (self.number_of_real_space_pixels as f32).sqrt();
    }

    /// C++ `Image::SetLogicalDimensions` (`image_trim.cpp:722`).
    pub fn set_logical_dimensions(
        &mut self,
        wanted_x_size: i32,
        wanted_y_size: i32,
        wanted_z_size: i32,
    ) {
        self.logical_x_dimension = wanted_x_size;
        self.logical_y_dimension = wanted_y_size;
        self.logical_z_dimension = wanted_z_size;
    }

    /// C++ `Image::UpdateLoopingAndAddressing` (`image_trim.cpp:731`).
    pub fn update_looping_and_addressing(&mut self) {
        self.physical_upper_bound_complex_x = self.logical_x_dimension / 2;
        self.physical_upper_bound_complex_y = self.logical_y_dimension - 1;
        self.physical_upper_bound_complex_z = self.logical_z_dimension - 1;

        self.update_physical_address_of_box_center();

        if is_even(self.logical_y_dimension) {
            self.physical_index_of_first_negative_frequency_y = self.logical_y_dimension / 2;
        } else {
            self.physical_index_of_first_negative_frequency_y = self.logical_y_dimension / 2 + 1;
        }

        if is_even(self.logical_z_dimension) {
            self.physical_index_of_first_negative_frequency_z = self.logical_z_dimension / 2;
        } else {
            self.physical_index_of_first_negative_frequency_z = self.logical_z_dimension / 2 + 1;
        }

        // Update the Fourier voxel size
        self.fourier_voxel_size_x = (1.0 / f64::from(self.logical_x_dimension)) as f32;
        self.fourier_voxel_size_y = (1.0 / f64::from(self.logical_y_dimension)) as f32;
        self.fourier_voxel_size_z = (1.0 / f64::from(self.logical_z_dimension)) as f32;

        // Logical bounds
        if is_even(self.logical_x_dimension) {
            self.logical_lower_bound_complex_x = -self.logical_x_dimension / 2;
            self.logical_upper_bound_complex_x = self.logical_x_dimension / 2;
            self.logical_lower_bound_real_x = -self.logical_x_dimension / 2;
            self.logical_upper_bound_real_x = self.logical_x_dimension / 2 - 1;
        } else {
            self.logical_lower_bound_complex_x = -(self.logical_x_dimension - 1) / 2;
            self.logical_upper_bound_complex_x = (self.logical_x_dimension - 1) / 2;
            self.logical_lower_bound_real_x = -(self.logical_x_dimension - 1) / 2;
            self.logical_upper_bound_real_x = (self.logical_x_dimension - 1) / 2;
        }

        if is_even(self.logical_y_dimension) {
            self.logical_lower_bound_complex_y = -self.logical_y_dimension / 2;
            self.logical_upper_bound_complex_y = self.logical_y_dimension / 2 - 1;
            self.logical_lower_bound_real_y = -self.logical_y_dimension / 2;
            self.logical_upper_bound_real_y = self.logical_y_dimension / 2 - 1;
        } else {
            self.logical_lower_bound_complex_y = -(self.logical_y_dimension - 1) / 2;
            self.logical_upper_bound_complex_y = (self.logical_y_dimension - 1) / 2;
            self.logical_lower_bound_real_y = -(self.logical_y_dimension - 1) / 2;
            self.logical_upper_bound_real_y = (self.logical_y_dimension - 1) / 2;
        }

        if is_even(self.logical_z_dimension) {
            self.logical_lower_bound_complex_z = -self.logical_z_dimension / 2;
            self.logical_upper_bound_complex_z = self.logical_z_dimension / 2 - 1;
            self.logical_lower_bound_real_z = -self.logical_z_dimension / 2;
            self.logical_upper_bound_real_z = self.logical_z_dimension / 2 - 1;
        } else {
            self.logical_lower_bound_complex_z = -(self.logical_z_dimension - 1) / 2;
            self.logical_upper_bound_complex_z = (self.logical_z_dimension - 1) / 2;
            self.logical_lower_bound_real_z = -(self.logical_z_dimension - 1) / 2;
            self.logical_upper_bound_real_z = (self.logical_z_dimension - 1) / 2;
        }
    }

    /// C++ `Image::UpdatePhysicalAddressOfBoxCenter` (`image_trim.cpp:817`).
    pub fn update_physical_address_of_box_center(&mut self) {
        self.physical_address_of_box_center_x = self.logical_x_dimension / 2;
        self.physical_address_of_box_center_y = self.logical_y_dimension / 2;
        self.physical_address_of_box_center_z = self.logical_z_dimension / 2;
    }

    /// C++ inline `Image::ReturnReal1DAddressFromPhysicalCoord` (`image.h:178`).
    pub fn return_real_1d_address_from_physical_coord(
        &self,
        wanted_x: i32,
        wanted_y: i32,
        wanted_z: i32,
    ) -> i64 {
        i64::from((self.logical_x_dimension + self.padding_jump_value) * self.logical_y_dimension)
            * i64::from(wanted_z)
            + i64::from(self.logical_x_dimension + self.padding_jump_value) * i64::from(wanted_y)
            + i64::from(wanted_x)
    }

    /// C++ inline `Image::ReturnRealPixelFromPhysicalCoord` (`image.h:186`).
    pub fn return_real_pixel_from_physical_coord(
        &self,
        wanted_x: i32,
        wanted_y: i32,
        wanted_z: i32,
    ) -> f32 {
        self.real_values
            [self.return_real_1d_address_from_physical_coord(wanted_x, wanted_y, wanted_z) as usize]
    }

    /// C++ inline `Image::ReturnFourier1DAddressFromPhysicalCoord` (`image.h:192`).
    pub fn return_fourier_1d_address_from_physical_coord(
        &self,
        wanted_x: i32,
        wanted_y: i32,
        wanted_z: i32,
    ) -> i64 {
        (i64::from(self.physical_upper_bound_complex_x + 1)
            * i64::from(self.physical_upper_bound_complex_y + 1))
            * i64::from(wanted_z)
            + i64::from(self.physical_upper_bound_complex_x + 1) * i64::from(wanted_y)
            + i64::from(wanted_x)
    }

    /// C++ inline `Image::ReturnFourier1DAddressFromLogicalCoord` (`image.h:198`).
    pub fn return_fourier_1d_address_from_logical_coord(
        &self,
        wanted_x: i32,
        wanted_y: i32,
        wanted_z: i32,
    ) -> i64 {
        let physical_x_address: i32;
        let physical_y_address: i32;
        let physical_z_address: i32;

        if wanted_x >= 0 {
            physical_x_address = wanted_x;

            if wanted_y >= 0 {
                physical_y_address = wanted_y;
            } else {
                physical_y_address = self.logical_y_dimension + wanted_y;
            }

            if wanted_z >= 0 {
                physical_z_address = wanted_z;
            } else {
                physical_z_address = self.logical_z_dimension + wanted_z;
            }
        } else {
            physical_x_address = -wanted_x;

            if wanted_y > 0 {
                physical_y_address = self.logical_y_dimension - wanted_y;
            } else {
                physical_y_address = -wanted_y;
            }

            if wanted_z > 0 {
                physical_z_address = self.logical_z_dimension - wanted_z;
            } else {
                physical_z_address = -wanted_z;
            }
        }

        self.return_fourier_1d_address_from_physical_coord(
            physical_x_address,
            physical_y_address,
            physical_z_address,
        )
    }

    /// C++ `Image::ReturnComplexPixelFromLogicalCoord` (`image.h:254`).
    pub fn return_complex_pixel_from_logical_coord(
        &self,
        wanted_x: i32,
        wanted_y: i32,
        wanted_z: i32,
        out_of_bounds_value: Complex,
    ) -> Complex {
        if wanted_x < self.logical_lower_bound_complex_x
            || wanted_x > self.logical_upper_bound_complex_x
            || wanted_y < self.logical_lower_bound_complex_y
            || wanted_y > self.logical_upper_bound_complex_y
            || wanted_z < self.logical_lower_bound_complex_z
            || wanted_z > self.logical_upper_bound_complex_z
        {
            out_of_bounds_value
        } else {
            self.complex_values(
                self.return_fourier_1d_address_from_logical_coord(wanted_x, wanted_y, wanted_z),
            )
        }
    }

    /// C++ inline `Image::ReturnVolumeInRealSpace` (`image.h:173`).
    pub fn return_volume_in_real_space(&self) -> i64 {
        self.number_of_real_space_pixels
    }

    /// C++ inline `Image::IsCubic` (`image.h:265`).
    pub fn is_cubic(&self) -> bool {
        self.logical_x_dimension == self.logical_y_dimension
            && self.logical_x_dimension == self.logical_z_dimension
    }

    /// C++ inline `Image::IsSquare` (`image.h:269`).
    pub fn is_square(&self) -> bool {
        self.logical_x_dimension == self.logical_y_dimension
    }

    /// C++ `Image::FourierComponentHasExplicitHermitianMate` (`image_trim.cpp:836`).
    pub fn fourier_component_has_explicit_hermitian_mate(
        &self,
        physical_index_x: i32,
        physical_index_y: i32,
        physical_index_z: i32,
    ) -> bool {
        let mut explicit_mate =
            physical_index_x == 0 && !(physical_index_y == 0 && physical_index_z == 0);

        // We assume that the Y dimension is the non-flat one
        if is_even(self.logical_y_dimension) {
            explicit_mate = explicit_mate
                && physical_index_y != self.physical_index_of_first_negative_frequency_y - 1;
        }

        if self.logical_z_dimension > 1 && is_even(self.logical_z_dimension) {
            explicit_mate = explicit_mate
                && physical_index_z != self.physical_index_of_first_negative_frequency_z - 1;
        }

        explicit_mate
    }

    /// C++ `Image::FourierComponentIsExplicitHermitianMate` (`image_trim.cpp:861`).
    pub fn fourier_component_is_explicit_hermitian_mate(
        &self,
        physical_index_x: i32,
        physical_index_y: i32,
        physical_index_z: i32,
    ) -> bool {
        physical_index_x == 0
            && (physical_index_y >= self.physical_index_of_first_negative_frequency_y
                || physical_index_z >= self.physical_index_of_first_negative_frequency_z)
    }

    /// C++ `Image::ForwardFFT` (`image_trim.cpp:883`).
    ///
    /// The whole body is inside the `/* IMOD ... */` comment: this library was
    /// trimmed of its FFTW dependency, so the call does nothing, does **not**
    /// set `is_in_real_space` and does not scale.  Nothing in `ctffind` calls
    /// it; substituting a real transform here would be a different program.
    pub fn forward_fft(&mut self, _should_scale: bool) {}

    /// C++ `Image::BackwardFFT` (`image_trim.cpp:903`).  Also empty — see
    /// [`Image::forward_fft`].
    pub fn backward_fft(&mut self) {}

    /// C++ `Image::DivideByConstant` (`image_trim.cpp:918`).
    pub fn divide_by_constant(&mut self, constant_to_divide_by: f32) {
        let inverse = (1. / f64::from(constant_to_divide_by)) as f32;
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] *= inverse;
        }
    }

    /// C++ `Image::MultiplyAddConstant` (`image_trim.cpp:929`).
    pub fn multiply_add_constant(&mut self, constant_to_multiply_by: f32, constant_to_add: f32) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] =
                self.real_values[pixel_counter] * constant_to_multiply_by + constant_to_add;
        }
    }

    /// C++ `Image::AddMultiplyConstant` (`image_trim.cpp:937`).
    pub fn add_multiply_constant(&mut self, constant_to_add: f32, constant_to_multiply_by: f32) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] =
                (self.real_values[pixel_counter] + constant_to_add) * constant_to_multiply_by;
        }
    }

    /// C++ `Image::AddMultiplyAddConstant` (`image_trim.cpp:945`).
    pub fn add_multiply_add_constant(
        &mut self,
        first_constant_to_add: f32,
        constant_to_multiply_by: f32,
        second_constant_to_add: f32,
    ) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] =
                (self.real_values[pixel_counter] + first_constant_to_add) * constant_to_multiply_by
                    + second_constant_to_add;
        }
    }

    /// C++ `Image::MultiplyByConstant` (`image_trim.cpp:956`).
    pub fn multiply_by_constant(&mut self, constant_to_multiply_by: f32) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] *= constant_to_multiply_by;
        }
    }

    /// C++ inline `Image::NormalizeFT` (`image.h:291`).
    pub fn normalize_ft(&mut self) {
        self.multiply_by_constant(self.ft_normalization_factor);
    }

    /// C++ inline `Image::NormalizeFTAndInvertRealValues` (`image.h:292`).
    pub fn normalize_ft_and_invert_real_values(&mut self) {
        self.multiply_by_constant(-self.ft_normalization_factor);
    }

    /// C++ `Image::TakeReciprocalRealValues` (`image_trim.cpp:966`).
    pub fn take_reciprocal_real_values(&mut self, zeros_become: f32) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            if self.real_values[pixel_counter] != 0.0 {
                self.real_values[pixel_counter] =
                    (1.0 / f64::from(self.real_values[pixel_counter])) as f32;
            } else {
                self.real_values[pixel_counter] = zeros_become;
            }
        }
    }

    /// C++ `Image::InvertRealValues` (`image_trim.cpp:978`).
    pub fn invert_real_values(&mut self) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] = -self.real_values[pixel_counter];
        }
    }

    /// C++ `Image::SquareRealValues` (`image_trim.cpp:988`).
    pub fn square_real_values(&mut self) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] *= self.real_values[pixel_counter];
        }
    }

    /// C++ `Image::ExponentiateRealValues` (`image_trim.cpp:998`).
    pub fn exponentiate_real_values(&mut self) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] = self.real_values[pixel_counter].exp();
        }
    }

    /// C++ `Image::SquareRootRealValues` (`image_trim.cpp:1008`).
    pub fn square_root_real_values(&mut self) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] = self.real_values[pixel_counter].sqrt();
        }
    }

    /// C++ `Image::IsConstant` (`image_trim.cpp:1019`).
    pub fn is_constant(&self) -> bool {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            if self.real_values[pixel_counter] != self.real_values[0] {
                return false;
            }
        }
        true
    }

    /// C++ `Image::IsBinary` (`image_trim.cpp:1029`).
    pub fn is_binary(&self) -> bool {
        let mut pixel_counter: i64 = 0;

        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    if self.real_values[pixel_counter as usize] != 0.0
                        && self.real_values[pixel_counter as usize] != 1.0
                    {
                        return false;
                    }
                    pixel_counter += 1;
                }
                pixel_counter += i64::from(self.padding_jump_value);
            }
        }

        true
    }

    /// C++ `Image::HasNan` (`image_trim.cpp:1052`).
    pub fn has_nan(&self) -> bool {
        let mut pixel_counter: i64 = 0;
        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    if self.real_values[pixel_counter as usize].is_nan() {
                        return true;
                    }
                    pixel_counter += 1;
                }
                pixel_counter += i64::from(self.padding_jump_value);
            }
        }
        false
    }

    /// C++ `Image::HasNegativeRealValue` (`image_trim.cpp:1078`).
    pub fn has_negative_real_value(&self) -> bool {
        let mut pixel_counter: i64 = 0;
        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    if self.real_values[pixel_counter as usize] < 0.0 {
                        return true;
                    }
                    pixel_counter += 1;
                }
                pixel_counter += i64::from(self.padding_jump_value);
            }
        }
        false
    }

    /// C++ `Image::QuickAndDirtyWriteSlice` (`image_trim.cpp:1288`), the IMOD
    /// replacement that hands an unpadded float slice to the registered writer.
    pub fn quick_and_dirty_write_slice(&self, filename: &str, _slice_to_write: i64) {
        let func = *S_SLICE_WRITE_FUNC
            .lock()
            .expect("slice write callback lock poisoned");
        let Some(func) = func else {
            return;
        };
        if self.padding_jump_value > 0 {
            let mut fdata =
                vec![0.0f32; (self.logical_x_dimension * self.logical_y_dimension) as usize];
            for iy in 0..self.logical_y_dimension {
                let dst = (iy * self.logical_x_dimension) as usize;
                let src = (iy * (self.logical_x_dimension + self.padding_jump_value)) as usize;
                fdata[dst..dst + self.logical_x_dimension as usize].copy_from_slice(
                    &self.real_values[src..src + self.logical_x_dimension as usize],
                );
            }
            func(
                filename,
                &fdata,
                self.logical_x_dimension,
                self.logical_y_dimension,
            );
        } else {
            func(
                filename,
                &self.real_values,
                self.logical_x_dimension,
                self.logical_y_dimension,
            );
        }
    }

    /// C++ `Image::AddFFTWPadding` (`image_trim.cpp:1310`).
    pub fn add_fftw_padding(&mut self) {
        let mut current_write_position: i64 =
            self.real_memory_allocated - i64::from(1 + self.padding_jump_value);
        let mut current_read_position: i64 = (i64::from(self.logical_x_dimension)
            * i64::from(self.logical_y_dimension)
            * i64::from(self.logical_z_dimension))
            - 1;

        for _z in 0..self.logical_z_dimension {
            for _y in 0..self.logical_y_dimension {
                for _x in 0..self.logical_x_dimension {
                    self.real_values[current_write_position as usize] =
                        self.real_values[current_read_position as usize];
                    current_write_position -= 1;
                    current_read_position -= 1;
                }

                current_write_position -= i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::RemoveFFTWPadding` (`image_trim.cpp:1337`).
    pub fn remove_fftw_padding(&mut self) {
        let mut current_write_position: i64 = 0;
        let mut current_read_position: i64 = 0;

        for _z in 0..self.logical_z_dimension {
            for _y in 0..self.logical_y_dimension {
                for _x in 0..self.logical_x_dimension {
                    self.real_values[current_write_position as usize] =
                        self.real_values[current_read_position as usize];
                    current_write_position += 1;
                    current_read_position += 1;
                }

                current_read_position += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::SetToConstant` (`image_trim.cpp:1362`).
    pub fn set_to_constant(&mut self, wanted_value: f32) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] = wanted_value;
        }
    }

    /// C++ `Image::AddConstant` (`image_trim.cpp:1372`).
    pub fn add_constant(&mut self, wanted_value: f32) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] += wanted_value;
        }
    }

    /// C++ `Image::SetMaximumValue` (`image_trim.cpp:1383`).
    pub fn set_maximum_value(&mut self, new_maximum_value: f32) {
        let mut address: i64 = 0;

        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    self.real_values[address as usize] =
                        if self.real_values[address as usize] < new_maximum_value {
                            self.real_values[address as usize]
                        } else {
                            new_maximum_value
                        };
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::SetMinimumValue` (`image_trim.cpp:1406`).
    pub fn set_minimum_value(&mut self, new_minimum_value: f32) {
        let mut address: i64 = 0;

        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    self.real_values[address as usize] =
                        if new_minimum_value < self.real_values[address as usize] {
                            self.real_values[address as usize]
                        } else {
                            new_minimum_value
                        };
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::Binarise` (`image_trim.cpp:1428`).
    pub fn binarise(&mut self, threshold_value: f32) {
        for address in 0..self.real_memory_allocated as usize {
            if self.real_values[address] >= threshold_value {
                self.real_values[address] = 1.0f32;
            } else {
                self.real_values[address] = 0.0f32;
            }
        }
    }

    /// C++ `Image::SetMinimumAndMaximumValues` (`image_trim.cpp:1442`).
    pub fn set_minimum_and_maximum_values(
        &mut self,
        new_minimum_value: f32,
        new_maximum_value: f32,
    ) {
        let mut address: i64 = 0;

        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    let clipped = if self.real_values[address as usize] < new_maximum_value {
                        self.real_values[address as usize]
                    } else {
                        new_maximum_value
                    };
                    self.real_values[address as usize] = if new_minimum_value < clipped {
                        clipped
                    } else {
                        new_minimum_value
                    };
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::ReturnMaximumDiagonalRadius` (`image_trim.cpp:1464`).
    pub fn return_maximum_diagonal_radius(&self) -> f32 {
        if self.is_in_real_space {
            // IMOD icl 11 add (double)
            (f64::from(self.physical_address_of_box_center_x).powi(2)
                + f64::from(self.physical_address_of_box_center_y).powi(2)
                + f64::from(self.physical_address_of_box_center_z).powi(2))
            .sqrt() as f32
        } else {
            (f64::from(self.logical_lower_bound_complex_x as f32 * self.fourier_voxel_size_x)
                .powi(2)
                + f64::from(self.logical_lower_bound_complex_y as f32 * self.fourier_voxel_size_y)
                    .powi(2)
                + f64::from(self.logical_lower_bound_complex_z as f32 * self.fourier_voxel_size_z)
                    .powi(2))
            .sqrt() as f32
        }
    }

    /// C++ `Image::GetMinMax` (`image_trim.cpp:1477`).
    pub fn get_min_max(&self, min_value: &mut f32, max_value: &mut f32) {
        *min_value = f32::MAX;
        *max_value = -f32::MAX;

        let mut address: i64 = 0;

        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    if self.real_values[address as usize] < *min_value {
                        *min_value = self.real_values[address as usize];
                    }
                    if self.real_values[address as usize] > *max_value {
                        *max_value = self.real_values[address as usize];
                    }

                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::ReturnAverageOfRealValuesOnEdges` (`image_trim.cpp:1505`).
    pub fn return_average_of_real_values_on_edges(&self) -> f32 {
        let mut sum: f64;
        let mut number_of_pixels: i64;
        let mut address: i64;

        sum = 0.0;
        number_of_pixels = 0;
        address = 0;

        if self.logical_z_dimension == 1 {
            // Two-dimensional image

            // First line
            for _pixel_counter in 0..self.logical_x_dimension {
                sum += f64::from(self.real_values[address as usize]);
                address += 1;
            }
            number_of_pixels += i64::from(self.logical_x_dimension);
            address += i64::from(self.padding_jump_value);
            // Other lines
            for _line_counter in 1..self.logical_y_dimension - 1 {
                sum += f64::from(self.real_values[address as usize]);
                address += i64::from(self.logical_x_dimension - 1);
                sum += f64::from(self.real_values[address as usize]);
                address += i64::from(self.padding_jump_value + 1);
                number_of_pixels += 2;
            }
            // Last line
            for _pixel_counter in 0..self.logical_x_dimension {
                sum += f64::from(self.real_values[address as usize]);
                address += 1;
            }
            number_of_pixels += i64::from(self.logical_x_dimension);
        } else {
            // Three-dimensional volume

            // First plane
            for _line_counter in 0..self.logical_y_dimension {
                for _pixel_counter in 0..self.logical_x_dimension {
                    sum += f64::from(self.real_values[address as usize]);
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
            number_of_pixels += i64::from(self.logical_x_dimension * self.logical_y_dimension);
            // Other planes
            for _plane_counter in 1..self.logical_z_dimension - 1 {
                for line_counter in 0..self.logical_y_dimension {
                    if line_counter == 0 || line_counter == self.logical_y_dimension - 1 {
                        // First and last line of that section
                        for _pixel_counter in 0..self.logical_x_dimension {
                            sum += f64::from(self.real_values[address as usize]);
                            address += 1;
                        }
                        address += i64::from(self.padding_jump_value);
                        number_of_pixels += i64::from(self.logical_x_dimension);
                    } else {
                        // All other lines (only count first and last pixel)
                        sum += f64::from(self.real_values[address as usize]);
                        address += i64::from(self.logical_x_dimension - 1);
                        sum += f64::from(self.real_values[address as usize]);
                        address += i64::from(self.padding_jump_value + 1);
                        number_of_pixels += 2;
                    }
                }
            }
            // Last plane
            for _line_counter in 0..self.logical_y_dimension {
                for _pixel_counter in 0..self.logical_x_dimension {
                    sum += f64::from(self.real_values[address as usize]);
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
            number_of_pixels += i64::from(self.logical_x_dimension * self.logical_y_dimension);
        }
        (sum / f64::from(number_of_pixels as f32)) as f32
    }

    /// C++ `Image::ReturnAverageOfRealValuesAtRadius` (`image_trim.cpp:1606`).
    pub fn return_average_of_real_values_at_radius(&self, wanted_mask_radius: f32) -> f32 {
        let mut sum = 0.0f64;
        let mut address: i64 = 0;
        let mut number_of_pixels: i64;
        let (mut x, mut y, mut z): (f32, f32, f32);
        let mask_radius_squared: f32;
        let mut distance_from_center_squared: f32;

        mask_radius_squared = wanted_mask_radius.powf(2.0);
        number_of_pixels = 0;
        for k in 0..self.logical_z_dimension {
            z = ((k - self.physical_address_of_box_center_z) as f32).powf(2.0);

            for j in 0..self.logical_y_dimension {
                y = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);

                for i in 0..self.logical_x_dimension {
                    x = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);

                    distance_from_center_squared = x + y + z;

                    if (distance_from_center_squared - mask_radius_squared).abs() < 4.0 {
                        sum += f64::from(self.real_values[address as usize]);
                        number_of_pixels += 1;
                    }
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }
        if number_of_pixels > 0 {
            (sum / number_of_pixels as f64) as f32
        } else {
            0.0
        }
    }

    /// C++ `Image::ReturnMaximumValue` (`image_trim.cpp:1660`).
    pub fn return_maximum_value(
        &self,
        minimum_distance_from_center: f32,
        minimum_distance_from_edge: f32,
    ) -> f32 {
        let (mut i_dist_from_center, mut j_dist_from_center, mut k_dist_from_center): (
            i32,
            i32,
            i32,
        );
        let mut maximum_value = -f32::MAX;
        let last_acceptable_address_x =
            (self.logical_x_dimension as f32 - minimum_distance_from_edge - 1.0) as i32;
        let last_acceptable_address_y =
            (self.logical_y_dimension as f32 - minimum_distance_from_edge - 1.0) as i32;
        let last_acceptable_address_z =
            (self.logical_z_dimension as f32 - minimum_distance_from_edge - 1.0) as i32;
        let mut address: i64 = 0;

        for k in 0..self.logical_z_dimension {
            if self.logical_z_dimension > 1 {
                k_dist_from_center = (k - self.physical_address_of_box_center_z).abs();
                if (k_dist_from_center as f32) < minimum_distance_from_center
                    || (k as f32) < minimum_distance_from_edge
                    || k > last_acceptable_address_z
                {
                    address += i64::from(
                        self.logical_y_dimension
                            * (self.logical_x_dimension + self.padding_jump_value),
                    );
                    continue;
                }
            }
            for j in 0..self.logical_y_dimension {
                j_dist_from_center = (j - self.physical_address_of_box_center_y).abs();
                if (j_dist_from_center as f32) < minimum_distance_from_center
                    || (j as f32) < minimum_distance_from_edge
                    || j > last_acceptable_address_y
                {
                    address += i64::from(self.logical_x_dimension + self.padding_jump_value);
                    continue;
                }
                for i in 0..self.logical_x_dimension {
                    i_dist_from_center = (i - self.physical_address_of_box_center_x).abs();
                    if (i_dist_from_center as f32) < minimum_distance_from_center
                        || (i as f32) < minimum_distance_from_edge
                        || i > last_acceptable_address_x
                    {
                        address += 1;
                        continue;
                    }

                    maximum_value = if maximum_value < self.real_values[address as usize] {
                        self.real_values[address as usize]
                    } else {
                        maximum_value
                    };
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }

        maximum_value
    }

    /// C++ `Image::ReturnMinimumValue` (`image_trim.cpp:1713`).
    pub fn return_minimum_value(
        &self,
        minimum_distance_from_center: f32,
        minimum_distance_from_edge: f32,
    ) -> f32 {
        let (mut i_dist_from_center, mut j_dist_from_center, mut k_dist_from_center): (
            i32,
            i32,
            i32,
        );
        let mut minimum_value = f32::MAX;
        let last_acceptable_address_x =
            (self.logical_x_dimension as f32 - minimum_distance_from_edge - 1.0) as i32;
        let last_acceptable_address_y =
            (self.logical_y_dimension as f32 - minimum_distance_from_edge - 1.0) as i32;
        let last_acceptable_address_z =
            (self.logical_z_dimension as f32 - minimum_distance_from_edge - 1.0) as i32;
        let mut address: i64 = 0;

        for k in 0..self.logical_z_dimension {
            if self.logical_z_dimension > 1 {
                k_dist_from_center = (k - self.physical_address_of_box_center_z).abs();
                if (k_dist_from_center as f32) < minimum_distance_from_center
                    || (k as f32) < minimum_distance_from_edge
                    || k > last_acceptable_address_z
                {
                    address += i64::from(
                        self.logical_y_dimension
                            * (self.logical_x_dimension + self.padding_jump_value),
                    );
                    continue;
                }
            }
            for j in 0..self.logical_y_dimension {
                j_dist_from_center = (j - self.physical_address_of_box_center_y).abs();
                if (j_dist_from_center as f32) < minimum_distance_from_center
                    || (j as f32) < minimum_distance_from_edge
                    || j > last_acceptable_address_y
                {
                    address += i64::from(self.logical_x_dimension + self.padding_jump_value);
                    continue;
                }
                for i in 0..self.logical_x_dimension {
                    i_dist_from_center = (i - self.physical_address_of_box_center_x).abs();
                    if (i_dist_from_center as f32) < minimum_distance_from_center
                        || (i as f32) < minimum_distance_from_edge
                        || i > last_acceptable_address_x
                    {
                        address += 1;
                        continue;
                    }

                    minimum_value = if self.real_values[address as usize] < minimum_value {
                        self.real_values[address as usize]
                    } else {
                        minimum_value
                    };
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }

        minimum_value
    }

    /// C++ `Image::ReturnMedianOfRealValues` (`image_trim.cpp:1765`).
    ///
    /// The source's `std::sort(buffer, buffer + number_of_voxels - 1)` leaves
    /// the final element out of the sort, which is kept here.
    pub fn return_median_of_real_values(&self) -> f32 {
        let number_of_voxels: i64 = i64::from(self.logical_x_dimension)
            * i64::from(self.logical_y_dimension)
            * i64::from(self.logical_z_dimension);
        let mut buffer_array = vec![0.0f32; number_of_voxels as usize];

        let median_value: f32;

        let mut address: i64 = 0;
        let mut buffer_counter: i64 = 0;

        for _k in 0..self.logical_z_dimension {
            for _j in 0..self.logical_y_dimension {
                for _i in 0..self.logical_x_dimension {
                    buffer_array[buffer_counter as usize] = self.real_values[address as usize];

                    buffer_counter += 1;
                    address += 1;
                }

                address += i64::from(self.padding_jump_value);
            }
        }

        let sorted_len = (number_of_voxels - 1).max(0) as usize;
        buffer_array[..sorted_len]
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        median_value = buffer_array[(number_of_voxels / 2) as usize];

        median_value
    }

    /// C++ `Image::ReturnAverageOfRealValues` (`image_trim.cpp:1803`).
    pub fn return_average_of_real_values(&self, wanted_mask_radius: f32, invert_mask: bool) -> f32 {
        let mut sum = 0.0f64;
        let mut address: i64 = 0;
        let mut number_of_pixels: i64;
        let (mut x, mut y, mut z): (f32, f32, f32);
        let mask_radius_squared: f32;
        let mut distance_from_center_squared: f32;

        if wanted_mask_radius > 0.0 {
            mask_radius_squared = wanted_mask_radius.powf(2.0);
            number_of_pixels = 0;
            for k in 0..self.logical_z_dimension {
                z = ((k - self.physical_address_of_box_center_z) as f32).powf(2.0);

                for j in 0..self.logical_y_dimension {
                    y = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);

                    for i in 0..self.logical_x_dimension {
                        x = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);

                        distance_from_center_squared = x + y + z;

                        if invert_mask {
                            if distance_from_center_squared > mask_radius_squared {
                                sum += f64::from(self.real_values[address as usize]);
                                number_of_pixels += 1;
                            }
                        } else if distance_from_center_squared <= mask_radius_squared {
                            sum += f64::from(self.real_values[address as usize]);
                            number_of_pixels += 1;
                        }
                        address += 1;
                    }
                    address += i64::from(self.padding_jump_value);
                }
            }
            if number_of_pixels > 0 {
                return (sum / number_of_pixels as f64) as f32;
            } else {
                return 0.0;
            }
        } else {
            for _k in 0..self.logical_z_dimension {
                for _j in 0..self.logical_y_dimension {
                    for _i in 0..self.logical_x_dimension {
                        sum += f64::from(self.real_values[address as usize]);
                        address += 1;
                    }
                    address += i64::from(self.padding_jump_value);
                }
            }
        }
        (sum / (i64::from(self.logical_x_dimension)
            * i64::from(self.logical_y_dimension)
            * i64::from(self.logical_z_dimension)) as f64) as f32
    }

    /// C++ `Image::ComputeAverageAndSigmaOfValuesInSpectrum` (`image_trim.cpp:1888`).
    pub fn compute_average_and_sigma_of_values_in_spectrum(
        &self,
        minimum_radius: f32,
        maximum_radius: f32,
        average: &mut f32,
        sigma: &mut f32,
        cross_half_width: i32,
    ) {
        let (mut x_sq, mut y_sq, mut rad_sq): (f32, f32, f32);
        let mut my_distribution = EmpiricalDistribution::new();
        let min_rad_sq = minimum_radius.powf(2.0);
        let max_rad_sq = maximum_radius.powf(2.0);
        let cross_half_width_sq = (cross_half_width as f32).powf(2.0);
        let mut address: i64 = -1;

        for j in 0..self.logical_y_dimension {
            y_sq = ((j - self.physical_address_of_box_center_y) as f32).powf(2.0);
            if y_sq <= cross_half_width_sq {
                address += i64::from(self.logical_x_dimension + self.padding_jump_value);
                continue;
            }
            for i in 0..self.logical_x_dimension {
                address += 1;
                x_sq = ((i - self.physical_address_of_box_center_x) as f32).powf(2.0);
                if x_sq <= cross_half_width_sq {
                    continue;
                }
                rad_sq = x_sq + y_sq;
                if rad_sq > min_rad_sq && rad_sq < max_rad_sq {
                    my_distribution.add_sample_value(self.real_values[address as usize]);
                }
            }
            address += i64::from(self.padding_jump_value);
        }
        *average = my_distribution.get_sample_mean();
        *sigma = my_distribution.get_sample_variance().sqrt();
    }

    /// C++ `Image::ZeroCentralPixel` (`image_trim.cpp:1931`).
    pub fn zero_central_pixel(&mut self) {
        if !self.is_in_real_space {
            self.set_complex_values(0, Complex::new(0.0, 0.0) * 1.0f32 + I * 0.0f32);
        } else {
            let mut address: i64 = 0;

            for j in 0..self.logical_y_dimension {
                for i in 0..self.logical_x_dimension {
                    if j == self.physical_address_of_box_center_y
                        && i == self.physical_address_of_box_center_x
                    {
                        self.real_values[address as usize] = 0.0;
                    }
                    address += 1;
                }
                address += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::SetMaximumValueOnCentralCross` (`image_trim.cpp:1961`).
    pub fn set_maximum_value_on_central_cross(&mut self, maximum_value: f32) {
        let mut address: i64 = 0;

        for j in 0..self.logical_y_dimension {
            for i in 0..self.logical_x_dimension {
                if j == self.physical_address_of_box_center_y
                    || i == self.physical_address_of_box_center_x
                {
                    self.real_values[address as usize] =
                        if self.real_values[address as usize] < maximum_value {
                            self.real_values[address as usize]
                        } else {
                            maximum_value
                        };
                }
                address += 1;
            }
            address += i64::from(self.padding_jump_value);
        }
    }

    /// C++ `Image::GetCorrelationWithCTF` (`image_trim.cpp:1986`).
    pub fn get_correlation_with_ctf(&self, ctf: Ctf) -> f32 {
        // Local variables
        let mut cross_product = 0.0f64;
        let mut norm_image = 0.0f64;
        let mut norm_ctf = 0.0f64;
        let mut number_of_values: i64 = 0;
        let (mut i_logi, mut j_logi): (f32, f32);
        let (mut i_logi_sq, mut j_logi_sq): (f32, f32);
        let inverse_logical_x_dimension = 1.0f32 / (self.logical_x_dimension as f32);
        let inverse_logical_y_dimension = 1.0f32 / (self.logical_y_dimension as f32);
        let mut current_spatial_frequency_squared: f32;
        let lowest_freq = ctf.get_lowest_frequency_for_fitting().powf(2.0);
        let highest_freq = ctf.get_highest_frequency_for_fitting().powf(2.0);
        let mut address: i64 = 0;
        let mut current_azimuth: f32;
        let mut current_ctf_value: f32;
        let central_cross_half_width: i32 = 10;
        let astigmatism_penalty: f32;

        // Loop over half of the image (ignore Friedel mates)
        for j in 0..self.logical_y_dimension {
            if j < self.physical_address_of_box_center_y - central_cross_half_width
                || j > self.physical_address_of_box_center_y + central_cross_half_width
            {
                address = i64::from(
                    j * (self.padding_jump_value + 2 * self.physical_address_of_box_center_x),
                );
                j_logi = ((j - self.physical_address_of_box_center_y) as f32)
                    * inverse_logical_y_dimension;
                j_logi_sq = j_logi.powf(2.0);
                for i in 0..self.physical_address_of_box_center_x - central_cross_half_width {
                    i_logi = ((i - self.physical_address_of_box_center_x) as f32)
                        * inverse_logical_x_dimension;
                    i_logi_sq = i_logi.powf(2.0);

                    // Where are we?
                    current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

                    if current_spatial_frequency_squared > lowest_freq
                        && current_spatial_frequency_squared < highest_freq
                    {
                        current_azimuth = j_logi.atan2(i_logi);
                        current_ctf_value = ctf
                            .evaluate(current_spatial_frequency_squared, current_azimuth)
                            .abs();
                        // accumulate results
                        number_of_values += 1;
                        cross_product +=
                            f64::from(self.real_values[(address + i64::from(i)) as usize])
                                * f64::from(current_ctf_value);
                        norm_image +=
                            f64::from(self.real_values[(address + i64::from(i)) as usize]).powi(2);
                        norm_ctf += f64::from(current_ctf_value).powi(2);
                    }
                }
            }
        }

        // Compute the penalty due to astigmatism
        if ctf.get_astigmatism_tolerance() > 0.0 {
            astigmatism_penalty = (f64::from(ctf.get_astigmatism().powf(2.0)) * 0.5
                / f64::from(ctf.get_astigmatism_tolerance().powf(2.0))
                / f64::from(number_of_values as f32)) as f32;
        } else {
            astigmatism_penalty = 0.0;
        }

        // The final score
        (cross_product / (norm_image * norm_ctf).sqrt() - f64::from(astigmatism_penalty)) as f32
    }

    /// C++ `Image::SetupQuickCorrelationWithCTF` (`image_trim.cpp:2066`).
    ///
    /// Called with `addresses` `None` it only counts the values that will be
    /// needed; called with the arrays it fills them and computes `norm_image`
    /// and `image_mean`, which do not change with the CTF.
    #[allow(clippy::too_many_arguments)]
    pub fn setup_quick_correlation_with_ctf(
        &self,
        ctf: Ctf,
        number_of_values: &mut i32,
        norm_image: &mut f64,
        image_mean: &mut f64,
        mut addresses: Option<&mut [i32]>,
        mut spatial_frequency_squared: Option<&mut [f32]>,
        mut azimuth: Option<&mut [f32]>,
    ) {
        // Local variables
        let (mut i_logi, mut j_logi): (f32, f32);
        let (mut i_logi_sq, mut j_logi_sq): (f32, f32);
        let inverse_logical_x_dimension = 1.0f32 / (self.logical_x_dimension as f32);
        let inverse_logical_y_dimension = 1.0f32 / (self.logical_y_dimension as f32);
        let mut current_spatial_frequency_squared: f32;

        let lowest_freq = ctf.get_lowest_frequency_for_fitting().powf(2.0);
        let highest_freq = ctf.get_highest_frequency_for_fitting().powf(2.0);
        let mut address: i32 = 0;
        let mut current_azimuth: f32;
        let central_cross_half_width: i32 = 10;
        let mut image_sum = 0.0f64;

        *number_of_values = 0;
        *norm_image = 0.0;
        *image_mean = 0.;

        // Loop over half of the image (ignore Friedel mates)
        for j in 0..self.logical_y_dimension {
            if j < self.physical_address_of_box_center_y - central_cross_half_width
                || j > self.physical_address_of_box_center_y + central_cross_half_width
            {
                address = j * (self.padding_jump_value + 2 * self.physical_address_of_box_center_x);
                j_logi = ((j - self.physical_address_of_box_center_y) as f32)
                    * inverse_logical_y_dimension;
                j_logi_sq = j_logi.powf(2.0);
                for i in 0..self.physical_address_of_box_center_x - central_cross_half_width {
                    i_logi = ((i - self.physical_address_of_box_center_x) as f32)
                        * inverse_logical_x_dimension;
                    i_logi_sq = i_logi.powf(2.0);

                    // Where are we?
                    current_spatial_frequency_squared = j_logi_sq + i_logi_sq;

                    if current_spatial_frequency_squared > lowest_freq
                        && current_spatial_frequency_squared < highest_freq
                    {
                        current_azimuth = j_logi.atan2(i_logi);
                        if let Some(addresses) = addresses.as_deref_mut() {
                            addresses[*number_of_values as usize] = address + i;
                            if let Some(sfs) = spatial_frequency_squared.as_deref_mut() {
                                sfs[*number_of_values as usize] = current_spatial_frequency_squared;
                            }
                            if let Some(az) = azimuth.as_deref_mut() {
                                az[*number_of_values as usize] = current_azimuth;
                            }
                            image_sum += f64::from(self.real_values[(address + i) as usize]);
                        }
                        *number_of_values += 1;
                    }
                }
            }
        }
        let _ = address;

        // Now get sum of squared deviations from mean, more accurate than using
        // raw cross-products
        if let Some(addresses) = addresses.as_deref_mut() {
            *image_mean = image_sum / f64::from(*number_of_values);
            for i in 0..*number_of_values as usize {
                *norm_image +=
                    (f64::from(self.real_values[addresses[i] as usize]) - *image_mean).powi(2);
            }
        }
    }

    /// C++ `Image::QuickCorrelationWithCTF` (`image_trim.cpp:2136`).
    #[allow(clippy::too_many_arguments)]
    pub fn quick_correlation_with_ctf(
        &self,
        ctf: Ctf,
        number_of_values: i32,
        norm_image: f64,
        image_mean: f64,
        addresses: &[i32],
        spatial_frequency_squared: &[f32],
        azimuth: &[f32],
    ) -> f32 {
        // Local variables
        let mut j: i32;
        let mut cross_product = 0.0f64;
        let mut norm_ctf = 0.0f64;
        let mut ctf_sum = 0.0f64;
        let mut current_ctf_value: f32;
        let astigmatism_penalty: f32;

        for i in 0..number_of_values as usize {
            j = addresses[i];
            current_ctf_value = (-ctf
                .phase_shift_given_squared_spatial_frequency_and_azimuth(
                    spatial_frequency_squared[i],
                    azimuth[i],
                )
                .sin())
            .abs();
            cross_product += f64::from(self.real_values[j as usize]) * f64::from(current_ctf_value);
            norm_ctf += f64::from(current_ctf_value).powi(2);
            ctf_sum += f64::from(current_ctf_value);
        }

        // Compute the penalty due to astigmatism
        if ctf.get_astigmatism_tolerance() > 0.0 {
            astigmatism_penalty = (f64::from(ctf.get_astigmatism().powf(2.0)) * 0.5
                / f64::from(ctf.get_astigmatism_tolerance().powf(2.0))
                / f64::from(number_of_values as f32)) as f32;
        } else {
            astigmatism_penalty = 0.0;
        }

        // The final score: norm_image is already a sum of squared deviations
        // from mean; norm_ctf requires adjustment to give true CC
        ((cross_product - image_mean * ctf_sum)
            / (norm_image * (norm_ctf - ctf_sum * ctf_sum / f64::from(number_of_values))).sqrt()
            - f64::from(astigmatism_penalty)) as f32
    }

    /// C++ `Image::ApplyMirrorAlongY` (`image_trim.cpp:2169`).
    pub fn apply_mirror_along_y(&mut self) {
        let mut address: i64 = i64::from(self.logical_x_dimension + self.padding_jump_value);
        let mut j_dist: i64;
        let mut temp_value: f32;

        for j in 1..self.physical_address_of_box_center_y {
            j_dist = i64::from(2 * (self.physical_address_of_box_center_y - j))
                * i64::from(self.logical_x_dimension + self.padding_jump_value);

            for _i in 0..self.logical_x_dimension {
                temp_value = self.real_values[address as usize];
                self.real_values[address as usize] = self.real_values[(address + j_dist) as usize];
                self.real_values[(address + j_dist) as usize] = temp_value;
                address += 1;
            }
            address += i64::from(self.padding_jump_value);
        }

        // The column j=0 is undefined, we set it to the average of the values
        // that were there before the mirror operation was applied
        temp_value = 0.0;
        for i in 0..self.logical_x_dimension as usize {
            temp_value += self.real_values[i];
        }
        temp_value /= self.logical_x_dimension as f32;
        for i in 0..self.logical_x_dimension as usize {
            self.real_values[i] = temp_value;
        }
    }

    /// C++ `Image::AddImage` (`image_trim.cpp:2205`).
    pub fn add_image(&mut self, other_image: &Image) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] += other_image.real_values[pixel_counter];
        }
    }

    /// C++ `Image::SubtractImage` (`image_trim.cpp:2216`).
    pub fn subtract_image(&mut self, other_image: &Image) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] -= other_image.real_values[pixel_counter];
        }
    }

    /// C++ `Image::SubtractSquaredImage` (`image_trim.cpp:2228`).
    pub fn subtract_squared_image(&mut self, other_image: &Image) {
        for pixel_counter in 0..self.real_memory_allocated as usize {
            self.real_values[pixel_counter] -= other_image.real_values[pixel_counter].powf(2.0);
        }
    }

    /// C++ `Image::ReturnFourierLogicalCoordGivenPhysicalCoord_X` (`image_trim.cpp:2239`).
    pub fn return_fourier_logical_coord_given_physical_coord_x(&self, physical_index: i32) -> i32 {
        if physical_index > self.physical_address_of_box_center_x {
            physical_index - self.logical_x_dimension
        } else {
            physical_index
        }
    }

    /// C++ `Image::ReturnFourierLogicalCoordGivenPhysicalCoord_Y` (`image_trim.cpp:2253`).
    pub fn return_fourier_logical_coord_given_physical_coord_y(&self, physical_index: i32) -> i32 {
        if physical_index >= self.physical_index_of_first_negative_frequency_y {
            physical_index - self.logical_y_dimension
        } else {
            physical_index
        }
    }

    /// C++ `Image::ReturnFourierLogicalCoordGivenPhysicalCoord_Z` (`image_trim.cpp:2265`).
    pub fn return_fourier_logical_coord_given_physical_coord_z(&self, physical_index: i32) -> i32 {
        if physical_index >= self.physical_index_of_first_negative_frequency_z {
            physical_index - self.logical_z_dimension
        } else {
            physical_index
        }
    }

    /// C++ `Image::Compute1DRotationalAverage` (`image_trim.cpp:2285`).
    pub fn compute_1d_rotational_average(
        &self,
        average: &mut Curve,
        number_of_values: &mut Curve,
        fractional_radius_in_real_space: bool,
    ) {
        let mut rad: f32;
        let mut address: i64;

        // Initialise
        average.zero_y_data();
        number_of_values.zero_y_data();
        address = 0;

        if self.is_in_real_space && !fractional_radius_in_real_space {
            let (mut i_logi, mut j_logi, mut k_logi): (i32, i32, i32);

            for k in 0..self.logical_z_dimension {
                // IMOD icl 11 switch to multiply 3 times
                k_logi = (k - self.physical_address_of_box_center_z)
                    * (k - self.physical_address_of_box_center_z);
                for j in 0..self.logical_y_dimension {
                    j_logi = (j - self.physical_address_of_box_center_y)
                        * (j - self.physical_address_of_box_center_y)
                        + k_logi;
                    for i in 0..self.logical_x_dimension {
                        i_logi = (i - self.physical_address_of_box_center_x)
                            * (i - self.physical_address_of_box_center_x)
                            + j_logi;
                        rad = (i_logi as f32).sqrt();
                        average.add_value_at_x_using_linear_interpolation(
                            rad,
                            self.real_values[address as usize],
                            true,
                        );
                        number_of_values.add_value_at_x_using_linear_interpolation(rad, 1.0, true);

                        // Increment the address
                        address += 1;
                    }
                    // End of the line in real space
                    address += i64::from(self.padding_jump_value);
                }
            }
        } else {
            let (mut i_logi, mut j_logi, mut k_logi): (f32, f32, f32);

            if self.is_in_real_space && fractional_radius_in_real_space {
                for k in 0..self.logical_z_dimension {
                    k_logi = f64::from(
                        (k - self.physical_address_of_box_center_z) as f32
                            * self.fourier_voxel_size_z,
                    )
                    .powi(2) as f32;
                    for j in 0..self.logical_y_dimension {
                        j_logi = (f64::from(
                            (j - self.physical_address_of_box_center_y) as f32
                                * self.fourier_voxel_size_y,
                        )
                        .powi(2)
                            + f64::from(k_logi)) as f32;
                        for i in 0..self.logical_x_dimension {
                            i_logi = (f64::from(
                                (i - self.physical_address_of_box_center_x) as f32
                                    * self.fourier_voxel_size_x,
                            )
                            .powi(2)
                                + f64::from(j_logi)) as f32;
                            rad = i_logi.sqrt();
                            average.add_value_at_x_using_linear_interpolation(
                                rad,
                                self.real_values[address as usize],
                                true,
                            );
                            number_of_values
                                .add_value_at_x_using_linear_interpolation(rad, 1.0, true);

                            address += 1;
                        }
                        address += i64::from(self.padding_jump_value);
                    }
                }
            } else {
                for k in 0..self.logical_z_dimension {
                    k_logi = f64::from(
                        self.return_fourier_logical_coord_given_physical_coord_z(k) as f32
                            * self.fourier_voxel_size_z,
                    )
                    .powi(2) as f32;
                    for j in 0..self.logical_y_dimension {
                        j_logi = (f64::from(
                            self.return_fourier_logical_coord_given_physical_coord_y(j) as f32
                                * self.fourier_voxel_size_y,
                        )
                        .powi(2)
                            + f64::from(k_logi)) as f32;
                        for i in 0..self.physical_upper_bound_complex_x {
                            i_logi = (f64::from(i as f32 * self.fourier_voxel_size_x).powi(2)
                                + f64::from(j_logi)) as f32;
                            if self.fourier_component_is_explicit_hermitian_mate(i, j, k) {
                                continue;
                            }
                            rad = i_logi.sqrt();
                            average.add_value_at_x_using_linear_interpolation(
                                rad,
                                self.complex_values(address).abs(),
                                true,
                            );
                            number_of_values
                                .add_value_at_x_using_linear_interpolation(rad, 1.0, true);

                            address += 1;
                        }
                    }
                }
            }
        }

        // Do the actual averaging
        for counter in 0..average.number_of_points as usize {
            if number_of_values.data_y[counter] != 0.0 {
                average.data_y[counter] /= number_of_values.data_y[counter];
            }
        }
    }

    /// C++ `Image::Compute1DPowerSpectrumCurve` (`image_trim.cpp:2398`).
    pub fn compute_1d_power_spectrum_curve(
        &self,
        curve_with_average_power: &mut Curve,
        curve_with_number_of_values: &mut Curve,
    ) {
        let (mut sq_dist_x, mut sq_dist_y, mut sq_dist_z): (f32, f32, f32);
        let mut address: i64;
        let mut spatial_frequency: f32;
        let mut number_of_hermitian_mates = 0i32;

        // Make sure the curves are clean
        curve_with_average_power.zero_y_data();
        curve_with_number_of_values.zero_y_data();

        // Get amplitudes and sum them into the curve object
        address = 0;
        for k in 0..=self.physical_upper_bound_complex_z {
            sq_dist_z = (self.return_fourier_logical_coord_given_physical_coord_z(k) as f32
                * self.fourier_voxel_size_z)
                .powf(2.0);
            for j in 0..=self.physical_upper_bound_complex_y {
                sq_dist_y = (self.return_fourier_logical_coord_given_physical_coord_y(j) as f32
                    * self.fourier_voxel_size_y)
                    .powf(2.0);
                for i in 0..=self.physical_upper_bound_complex_x {
                    if self.fourier_component_is_explicit_hermitian_mate(i, j, k) {
                        number_of_hermitian_mates += 1;
                        address += 1;
                        continue;
                    } else {
                        sq_dist_x = (i as f32 * self.fourier_voxel_size_x).powf(2.0);
                        spatial_frequency = (sq_dist_x + sq_dist_y + sq_dist_z).sqrt();

                        let value = self.complex_values(address);
                        curve_with_average_power.add_value_at_x_using_linear_interpolation(
                            spatial_frequency,
                            value.re * value.re + value.im * value.im,
                            true,
                        );
                        curve_with_number_of_values.add_value_at_x_using_linear_interpolation(
                            spatial_frequency,
                            1.0,
                            true,
                        );

                        address += 1;
                    }
                }
            }
        }
        let _ = number_of_hermitian_mates;

        // Do the actual averaging
        for counter in 0..curve_with_average_power.number_of_points as usize {
            if curve_with_number_of_values.data_y[counter] > 0.0 {
                curve_with_average_power.data_y[counter] /=
                    curve_with_number_of_values.data_y[counter];
            } else {
                curve_with_average_power.data_y[counter] = 0.0;
            }
        }
    }

    /// C++ `Image::ComputeAmplitudeSpectrumFull2D` (`image_trim.cpp:2470`).
    pub fn compute_amplitude_spectrum_full_2d(&self, amplitude_spectrum: &mut Image) {
        let mut address_in_amplitude_spectrum: i64 = 0;
        let mut address_in_self: i64;

        // Loop over the amplitude spectrum
        for ampl_addr_j in 0..amplitude_spectrum.logical_y_dimension {
            for ampl_addr_i in 0..amplitude_spectrum.logical_x_dimension {
                address_in_self = self.return_fourier_1d_address_from_logical_coord(
                    ampl_addr_i - amplitude_spectrum.physical_address_of_box_center_x,
                    ampl_addr_j - amplitude_spectrum.physical_address_of_box_center_y,
                    0,
                );
                amplitude_spectrum.real_values[address_in_amplitude_spectrum as usize] =
                    self.complex_values(address_in_self).abs();
                address_in_amplitude_spectrum += 1;
            }
            address_in_amplitude_spectrum += i64::from(amplitude_spectrum.padding_jump_value);
        }

        // Done
        amplitude_spectrum.is_in_real_space = true;
        amplitude_spectrum.object_is_centred_in_box = true;
    }

    /// C++ `Image::SpectrumBoxConvolution` (`image_trim.cpp:2512`).
    ///
    /// Real-space box convolution meant for 2D amplitude spectra, adapted from
    /// CTFFIND3's MSMOOTH with a different wrap-around behaviour; DNM rewrote
    /// it around per-line sums, which is what is translated here.
    pub fn spectrum_box_convolution(
        &self,
        output_image: &mut Image,
        box_size: i32,
        minimum_radius: f32,
    ) {
        // Variables
        let half_box_size = (box_size - 1) / 2;
        let cross_half_width_to_ignore = 1;
        let mut i_sq: i32;
        let mut j_sq: i32;
        let mut jj: i32;
        let mut num_voxels: i32;
        let minimum_radius_sq = f64::from(minimum_radius).powi(2) as f32;
        let mut radius_sq: f32;
        let first_i_to_ignore = self.physical_address_of_box_center_x - cross_half_width_to_ignore;
        let last_i_to_ignore = self.physical_address_of_box_center_x + cross_half_width_to_ignore;
        let first_j_to_ignore = self.physical_address_of_box_center_y - cross_half_width_to_ignore;
        let last_j_to_ignore = self.physical_address_of_box_center_y + cross_half_width_to_ignore;

        // Addresses
        let mut address_within_output: i64 = 0;

        // Starting and ending x indexes of one or two loops for each line
        let mut x1start = vec![0i32; self.logical_x_dimension as usize];
        let mut x1end = vec![0i32; self.logical_x_dimension as usize];
        let mut x2start = vec![0i32; self.logical_x_dimension as usize];
        let mut x2end = vec![0i32; self.logical_x_dimension as usize];
        let mut num_in_line_sum = vec![0i32; self.logical_x_dimension as usize];
        let mut line_sums =
            vec![0.0f32; (self.logical_x_dimension * self.logical_y_dimension) as usize];
        let mut sum: f32;
        let mut ybase: i32;

        // Get the limits for one or two loops for making line sums at each X position
        for i in 0..self.logical_x_dimension as usize {
            x1start[i] = i as i32 - half_box_size;
            x1end[i] = i as i32 + half_box_size;
            x2start[i] = 0;
            x2end[i] = -1;

            // Wrap around left edge
            if x1start[i] < 0 {
                x2start[i] = x1start[i] + self.logical_x_dimension;
                x2end[i] = self.logical_x_dimension - 1;
                x1start[i] = 0;
            }
            // Or wrap around right edge
            else if x1end[i] >= self.logical_x_dimension {
                x2end[i] = x1end[i] - self.logical_x_dimension;
                x2start[i] = 0;
                x1end[i] = self.logical_x_dimension - 1;
            }
            // Or handle intersection with the central cross by trimming or
            // splitting into two loops
            else if x1start[i] <= last_i_to_ignore && x1end[i] >= first_i_to_ignore {
                if x1start[i] >= first_i_to_ignore {
                    x1start[i] = last_i_to_ignore + 1;
                } else if x1end[i] <= last_i_to_ignore {
                    x1end[i] = first_i_to_ignore - 1;
                } else {
                    x2end[i] = x1end[i];
                    x2start[i] = last_i_to_ignore + 1;
                    x1end[i] = first_i_to_ignore - 1;
                }
            }
            num_in_line_sum[i] = x1end[i] + 1 - x1start[i];
            if x2end[i] >= x2start[i] {
                num_in_line_sum[i] += x2end[i] + 1 - x2start[i];
            }
        }

        // Loop over Y positions for line sums
        for jj in 0..self.logical_y_dimension {
            ybase = jj * (self.logical_x_dimension + self.padding_jump_value);

            // Form line sums at each X position
            for i in 0..self.logical_x_dimension as usize {
                sum = 0.;
                for ii in x1start[i]..=x1end[i] {
                    sum += self.real_values[(ii + ybase) as usize];
                }
                for ii in x2start[i]..=x2end[i] {
                    sum += self.real_values[(ii + ybase) as usize];
                }
                line_sums[i + (jj * self.logical_x_dimension) as usize] = sum;
            }
        }

        // Loop over the output image
        for j in 0..self.logical_y_dimension {
            j_sq = (j - self.physical_address_of_box_center_y)
                * (j - self.physical_address_of_box_center_y);

            for i in 0..self.logical_x_dimension {
                i_sq = (i - self.physical_address_of_box_center_x)
                    * (i - self.physical_address_of_box_center_x);

                radius_sq = (i_sq + j_sq) as f32;

                if radius_sq <= minimum_radius_sq {
                    output_image.real_values[address_within_output as usize] =
                        self.real_values[address_within_output as usize];
                } else {
                    output_image.real_values[address_within_output as usize] = 0.0e0;
                    num_voxels = 0;

                    // Loop over the lines to sum at this pixel to get the box sum
                    for m in -half_box_size..=half_box_size {
                        jj = j + m;
                        // wrap around
                        if jj < 0 {
                            jj += self.logical_y_dimension;
                        }
                        if jj >= self.logical_y_dimension {
                            jj -= self.logical_y_dimension;
                        }

                        // In central cross?
                        if jj >= first_j_to_ignore && jj <= last_j_to_ignore {
                            continue;
                        }

                        output_image.real_values[address_within_output as usize] +=
                            line_sums[(i + jj * self.logical_x_dimension) as usize];
                        num_voxels += num_in_line_sum[i as usize];
                    } // end of loop over the box

                    if num_voxels == 0 {
                        output_image.real_values[address_within_output as usize] =
                            self.real_values[address_within_output as usize];
                    } else {
                        output_image.real_values[address_within_output as usize] /=
                            num_voxels as f32;
                    }
                }

                address_within_output += 1;
            }
            address_within_output += i64::from(output_image.padding_jump_value);
        }
    }

    /// C++ `Image::ClipIntoLargerRealSpace2D` (`image_trim.cpp:3100`).
    pub fn clip_into_larger_real_space_2d(
        &self,
        other_image: &mut Image,
        wanted_padding_value: f32,
    ) {
        other_image.is_in_real_space = self.is_in_real_space;
        other_image.object_is_centred_in_box = self.object_is_centred_in_box;

        // Looping variables
        let mut address_in_self: i64 = 0;
        let mut address_in_other: i64 = 0;

        let i_lower_bound =
            other_image.physical_address_of_box_center_x - self.physical_address_of_box_center_x;
        let j_lower_bound =
            other_image.physical_address_of_box_center_y - self.physical_address_of_box_center_y;
        let i_upper_bound = i_lower_bound + self.logical_x_dimension - 1;
        let j_upper_bound = j_lower_bound + self.logical_y_dimension - 1;

        // Loop over the other (larger) image
        for j in 0..other_image.logical_y_dimension {
            // Check whether this line is outside of the original image
            if j < j_lower_bound || j > j_upper_bound {
                // Fill this line with the padding value
                for _i in 0..other_image.logical_x_dimension {
                    other_image.real_values[address_in_other as usize] = wanted_padding_value;
                    address_in_other += 1;
                }
            } else {
                // This line is within the central region
                for i in 0..other_image.logical_x_dimension {
                    if i < i_lower_bound || i > i_upper_bound {
                        // We are near the beginning or the end of the line
                        other_image.real_values[address_in_other as usize] = wanted_padding_value;
                    } else {
                        other_image.real_values[address_in_other as usize] =
                            self.real_values[address_in_self as usize];
                        address_in_self += 1;
                    }
                    address_in_other += 1;
                }
            }
            // We've reached the end of the line
            address_in_other += i64::from(other_image.padding_jump_value);
            if j >= j_lower_bound {
                address_in_self += i64::from(self.padding_jump_value);
            }
        }
    }

    /// C++ `Image::ClipInto` (`image_trim.cpp:3169`).
    #[allow(clippy::too_many_arguments)]
    pub fn clip_into(
        &self,
        other_image: &mut Image,
        wanted_padding_value: f32,
        fill_with_noise: bool,
        wanted_noise_sigma: f32,
        wanted_coordinate_of_box_center_x: i32,
        wanted_coordinate_of_box_center_y: i32,
        wanted_coordinate_of_box_center_z: i32,
    ) {
        let mut pixel_counter: i64 = 0;

        let (mut temp_logical_x, mut temp_logical_y, mut temp_logical_z): (i32, i32, i32);

        let (mut k, mut kk_logi): (i32, i32);
        let (mut j, mut jj_logi): (i32, i32);
        let (mut i, mut ii_logi): (i32, i32);

        // take other following attributes
        other_image.is_in_real_space = self.is_in_real_space;
        other_image.object_is_centred_in_box = self.object_is_centred_in_box;

        if self.is_in_real_space {
            for kk in 0..other_image.logical_z_dimension {
                kk_logi = kk - other_image.physical_address_of_box_center_z;
                k = self.physical_address_of_box_center_z
                    + wanted_coordinate_of_box_center_z
                    + kk_logi;

                for jj in 0..other_image.logical_y_dimension {
                    jj_logi = jj - other_image.physical_address_of_box_center_y;
                    j = self.physical_address_of_box_center_y
                        + wanted_coordinate_of_box_center_y
                        + jj_logi;

                    for ii in 0..other_image.logical_x_dimension {
                        ii_logi = ii - other_image.physical_address_of_box_center_x;
                        i = self.physical_address_of_box_center_x
                            + wanted_coordinate_of_box_center_x
                            + ii_logi;

                        if k < 0
                            || k >= self.logical_z_dimension
                            || j < 0
                            || j >= self.logical_y_dimension
                            || i < 0
                            || i >= self.logical_x_dimension
                        {
                            other_image.real_values[pixel_counter as usize] = wanted_padding_value;
                        } else {
                            other_image.real_values[pixel_counter as usize] =
                                self.return_real_pixel_from_physical_coord(i, j, k);
                        }

                        pixel_counter += 1;
                    }

                    pixel_counter += i64::from(other_image.padding_jump_value);
                }
            }
        } else {
            for kk in 0..=other_image.physical_upper_bound_complex_z {
                temp_logical_z =
                    other_image.return_fourier_logical_coord_given_physical_coord_z(kk);

                for jj in 0..=other_image.physical_upper_bound_complex_y {
                    temp_logical_y =
                        other_image.return_fourier_logical_coord_given_physical_coord_y(jj);

                    for ii in 0..=other_image.physical_upper_bound_complex_x {
                        temp_logical_x = ii;

                        if !fill_with_noise {
                            let value = self.return_complex_pixel_from_logical_coord(
                                temp_logical_x,
                                temp_logical_y,
                                temp_logical_z,
                                Complex::new(wanted_padding_value, 0.0) + I * 0.0f32,
                            );
                            other_image.set_complex_values(pixel_counter, value);
                        } else if temp_logical_x < self.logical_lower_bound_complex_x
                            || temp_logical_x > self.logical_upper_bound_complex_x
                            || temp_logical_y < self.logical_lower_bound_complex_y
                            || temp_logical_y > self.logical_upper_bound_complex_y
                            || temp_logical_z < self.logical_lower_bound_complex_z
                            || temp_logical_z > self.logical_upper_bound_complex_z
                        {
                            let mut generator = global_random_number_generator();
                            let re = generator.get_normal_random() * wanted_noise_sigma;
                            let im = generator.get_normal_random() * wanted_noise_sigma;
                            drop(generator);
                            other_image
                                .set_complex_values(pixel_counter, Complex::new(re, 0.0) + I * im);
                        } else {
                            let value = self.complex_values(
                                self.return_fourier_1d_address_from_logical_coord(
                                    temp_logical_x,
                                    temp_logical_y,
                                    temp_logical_z,
                                ),
                            );
                            other_image.set_complex_values(pixel_counter, value);
                        }
                        pixel_counter += 1;
                    }
                }
            }

            // When we are clipping into a larger volume in Fourier space, there
            // is a half-plane (vol) or half-line (2D image) at Nyquist for which
            // FFTW does not explicitly tell us the values. We need to fill them in.
            if self.logical_y_dimension < other_image.logical_y_dimension
                || self.logical_z_dimension < other_image.logical_z_dimension
            {
                // For a 2D image
                if self.logical_z_dimension == 1 {
                    let jj = self.physical_index_of_first_negative_frequency_y;
                    for ii in 0..=self.physical_upper_bound_complex_x {
                        let value = self.complex_values(
                            self.return_fourier_1d_address_from_physical_coord(ii, jj, 0),
                        );
                        let target =
                            other_image.return_fourier_1d_address_from_physical_coord(ii, jj, 0);
                        other_image.set_complex_values(target, value);
                    }
                }
                // For a 3D volume
                else {
                    // Deal with the positive Nyquist of the 2nd dimension
                    for kk_logi in
                        self.logical_lower_bound_complex_z..=self.logical_upper_bound_complex_z
                    {
                        let jj = self.physical_index_of_first_negative_frequency_y;
                        let jj_logi = self.logical_lower_bound_complex_y;
                        for ii in 0..=self.physical_upper_bound_complex_x {
                            let value = self.complex_values(
                                self.return_fourier_1d_address_from_logical_coord(
                                    ii, jj_logi, kk_logi,
                                ),
                            );
                            let target = other_image
                                .return_fourier_1d_address_from_logical_coord(ii, jj, kk_logi);
                            other_image.set_complex_values(target, value);
                        }
                    }

                    // Deal with the positive Nyquist in the 3rd dimension
                    let kk = self.physical_index_of_first_negative_frequency_z;
                    let kk_mirror = other_image.logical_z_dimension
                        - self.physical_index_of_first_negative_frequency_z;
                    let mut jj_mirror: i32;
                    for jj in 1..=self.physical_index_of_first_negative_frequency_y {
                        jj_mirror = jj;
                        for ii in 0..=self.physical_upper_bound_complex_x {
                            let source = other_image.return_fourier_1d_address_from_physical_coord(
                                ii, jj_mirror, kk_mirror,
                            );
                            let value = other_image.complex_values(source);
                            let target = other_image
                                .return_fourier_1d_address_from_physical_coord(ii, jj, kk);
                            other_image.set_complex_values(target, value);
                        }
                    }
                    for jj in other_image.logical_y_dimension
                        - self.physical_index_of_first_negative_frequency_y
                        ..=other_image.logical_y_dimension - 1
                    {
                        jj_mirror = jj;
                        for ii in 0..=self.physical_upper_bound_complex_x {
                            let source = other_image.return_fourier_1d_address_from_physical_coord(
                                ii, jj_mirror, kk_mirror,
                            );
                            let value = other_image.complex_values(source);
                            let target = other_image
                                .return_fourier_1d_address_from_physical_coord(ii, jj, kk);
                            other_image.set_complex_values(target, value);
                        }
                    }
                    let jj = 0;
                    for ii in 0..=self.physical_upper_bound_complex_x {
                        let source = other_image
                            .return_fourier_1d_address_from_physical_coord(ii, jj, kk_mirror);
                        let value = other_image.complex_values(source);
                        let target =
                            other_image.return_fourier_1d_address_from_physical_coord(ii, jj, kk);
                        other_image.set_complex_values(target, value);
                    }
                }
            }
        }
    }

    /// C++ `Image::GetRealValueByLinearInterpolationNoBoundsCheckImage`
    /// (`image_trim.cpp:3354`).
    pub fn get_real_value_by_linear_interpolation_no_bounds_check_image(
        &self,
        x: f32,
        y: f32,
        interpolated_value: &mut f32,
    ) {
        let i_start = x as i32;
        let j_start = y as i32;
        let x_dist = x - (i_start as f32);
        let y_dist = y - (j_start as f32);
        let x_dist_m = (1.0 - f64::from(x_dist)) as f32;
        let y_dist_m = (1.0 - f64::from(y_dist)) as f32;

        let address_1 = j_start * (self.logical_x_dimension + self.padding_jump_value) + i_start;
        let address_2 = address_1 + self.logical_x_dimension + self.padding_jump_value;

        *interpolated_value = x_dist_m * y_dist_m * self.real_values[address_1 as usize]
            + x_dist * y_dist_m * self.real_values[(address_1 + 1) as usize]
            + x_dist_m * y_dist * self.real_values[address_2 as usize]
            + x_dist * y_dist * self.real_values[(address_2 + 1) as usize];
    }

    /// C++ `Image::Resize` (`image_trim.cpp:3381`).
    pub fn resize(
        &mut self,
        wanted_x_dimension: i32,
        wanted_y_dimension: i32,
        wanted_z_dimension: i32,
        wanted_padding_value: f32,
    ) {
        if self.logical_x_dimension == wanted_x_dimension
            && self.logical_y_dimension == wanted_y_dimension
            && self.logical_z_dimension == wanted_z_dimension
        {
            return;
        }

        let mut temp_image = Image::new();

        temp_image.allocate(
            wanted_x_dimension,
            wanted_y_dimension,
            wanted_z_dimension,
            self.is_in_real_space,
        );
        self.clip_into(&mut temp_image, wanted_padding_value, false, 1.0, 0, 0, 0);

        self.consume(&mut temp_image);
    }

    /// C++ `Image::CopyFrom` (`image_trim.cpp:3398`).
    pub fn copy_from(&mut self, other_image: &Image) {
        self.assign(other_image);
    }

    /// C++ `Image::CopyLoopingAndAddressingFrom` (`image_trim.cpp:3403`).
    pub fn copy_looping_and_addressing_from(&mut self, other_image: &Image) {
        self.object_is_centred_in_box = other_image.object_is_centred_in_box;
        self.logical_x_dimension = other_image.logical_x_dimension;
        self.logical_y_dimension = other_image.logical_y_dimension;
        self.logical_z_dimension = other_image.logical_z_dimension;

        self.physical_upper_bound_complex_x = other_image.physical_upper_bound_complex_x;
        self.physical_upper_bound_complex_y = other_image.physical_upper_bound_complex_y;
        self.physical_upper_bound_complex_z = other_image.physical_upper_bound_complex_z;

        self.physical_address_of_box_center_x = other_image.physical_address_of_box_center_x;
        self.physical_address_of_box_center_y = other_image.physical_address_of_box_center_y;
        self.physical_address_of_box_center_z = other_image.physical_address_of_box_center_z;

        self.physical_index_of_first_negative_frequency_y =
            other_image.physical_index_of_first_negative_frequency_y;
        self.physical_index_of_first_negative_frequency_z =
            other_image.physical_index_of_first_negative_frequency_z;

        self.fourier_voxel_size_x = other_image.fourier_voxel_size_x;
        self.fourier_voxel_size_y = other_image.fourier_voxel_size_y;
        self.fourier_voxel_size_z = other_image.fourier_voxel_size_z;

        self.logical_upper_bound_complex_x = other_image.logical_upper_bound_complex_x;
        self.logical_upper_bound_complex_y = other_image.logical_upper_bound_complex_y;
        self.logical_upper_bound_complex_z = other_image.logical_upper_bound_complex_z;

        self.logical_lower_bound_complex_x = other_image.logical_lower_bound_complex_x;
        self.logical_lower_bound_complex_y = other_image.logical_lower_bound_complex_y;
        self.logical_lower_bound_complex_z = other_image.logical_lower_bound_complex_z;

        // Note: the source copies the *complex* bounds into the real ones here.
        self.logical_upper_bound_real_x = other_image.logical_upper_bound_complex_x;
        self.logical_upper_bound_real_y = other_image.logical_upper_bound_complex_y;
        self.logical_upper_bound_real_z = other_image.logical_upper_bound_complex_z;

        self.logical_lower_bound_real_x = other_image.logical_lower_bound_complex_x;
        self.logical_lower_bound_real_y = other_image.logical_lower_bound_complex_y;
        self.logical_lower_bound_real_z = other_image.logical_lower_bound_complex_z;

        self.padding_jump_value = other_image.padding_jump_value;
    }

    /// C++ `Image::Consume` (`image_trim.cpp:3445`): copy the parameters then
    /// steal the memory of another image, leaving it an empty shell.
    pub fn consume(&mut self, other_image: &mut Image) {
        if self.is_in_memory {
            self.deallocate();
        }

        self.is_in_real_space = other_image.is_in_real_space;
        self.real_memory_allocated = other_image.real_memory_allocated;
        self.copy_looping_and_addressing_from(other_image);

        self.real_values = std::mem::take(&mut other_image.real_values);
        self.is_in_memory = other_image.is_in_memory;

        other_image.is_in_memory = false;

        self.number_of_real_space_pixels = other_image.number_of_real_space_pixels;
        self.ft_normalization_factor = other_image.ft_normalization_factor;
    }

    /// C++ `Image::HasSameDimensionsAs` (`image_trim.cpp:3667`).
    pub fn has_same_dimensions_as(&self, other_image: &Image) -> bool {
        self.logical_x_dimension == other_image.logical_x_dimension
            && self.logical_y_dimension == other_image.logical_y_dimension
            && self.logical_z_dimension == other_image.logical_z_dimension
    }

    /// C++ `Image::ReturnLinearInterpolated2D` (`image_trim.cpp:3676`).
    pub fn return_linear_interpolated_2d(
        &self,
        wanted_physical_x_coordinate: f32,
        wanted_physical_y_coordinate: f32,
    ) -> f32 {
        if wanted_physical_x_coordinate < 0.0
            || wanted_physical_x_coordinate > (self.logical_x_dimension - 1) as f32
        {
            return 0.0;
        }
        if wanted_physical_y_coordinate < 0.0
            || wanted_physical_y_coordinate > (self.logical_y_dimension - 1) as f32
        {
            return 0.0;
        }

        let int_x_coordinate: i32;
        let int_y_coordinate: i32;
        let mut int_x_coordinate1: i32;
        let mut int_y_coordinate1: i32;
        let mut int_y: i32;

        let mut weight_x: f32;
        let mut weight_y: f32;

        let mut sum = 0.0f32;

        int_x_coordinate = wanted_physical_x_coordinate.floor() as i32;
        int_y_coordinate = wanted_physical_y_coordinate.floor() as i32;
        int_x_coordinate1 = int_x_coordinate + 1;
        int_y_coordinate1 = int_y_coordinate + 1;
        int_x_coordinate1 = int_x_coordinate1.min(self.logical_x_dimension - 1);
        int_y_coordinate1 = int_y_coordinate1.min(self.logical_y_dimension - 1);

        for j in int_y_coordinate..=int_y_coordinate1 {
            weight_y = (1.0 - f64::from((wanted_physical_y_coordinate - j as f32).abs())) as f32;
            int_y = (self.logical_x_dimension + self.padding_jump_value) * j;
            for i in int_x_coordinate..=int_x_coordinate1 {
                weight_x =
                    (1.0 - f64::from((wanted_physical_x_coordinate - i as f32).abs())) as f32;
                sum += self.real_values[(int_y + i) as usize] * weight_x * weight_y;
            }
        }

        sum
    }

    /// C++ `Image::CorrectMagnificationDistortion` (`image_trim.cpp:3717`).
    pub fn correct_magnification_distortion(
        &mut self,
        distortion_angle: f32,
        distortion_major_axis: f32,
        distortion_minor_axis: f32,
    ) {
        let mut pixel_counter: i64 = 0;
        let angle_in_radians = super::functions::deg_2_rad(distortion_angle);

        let x_scale_factor = (1.0 / f64::from(distortion_major_axis)) as f32;
        let y_scale_factor = (1.0 / f64::from(distortion_minor_axis)) as f32;

        let average_edge_value = self.return_average_of_real_values_on_edges();

        let mut new_x: f32;
        let mut new_y: f32;

        let mut final_x: f32;
        let mut final_y: f32;

        let mut buffer_image = Image::new();
        buffer_image.allocate(
            self.logical_x_dimension,
            self.logical_y_dimension,
            self.logical_z_dimension,
            self.is_in_real_space,
        );

        for y in 0..self.logical_y_dimension {
            for x in 0..self.logical_x_dimension {
                // first rotation
                new_x = ((y - self.physical_address_of_box_center_y) as f32)
                    * (-angle_in_radians).sin()
                    + ((x - self.physical_address_of_box_center_x) as f32)
                        * (-angle_in_radians).cos();
                new_y = ((y - self.physical_address_of_box_center_y) as f32)
                    * (-angle_in_radians).cos()
                    - ((x - self.physical_address_of_box_center_x) as f32)
                        * (-angle_in_radians).sin();

                // scale factor
                new_x *= x_scale_factor;
                new_y *= y_scale_factor;

                new_x += self.physical_address_of_box_center_x as f32;
                new_y += self.physical_address_of_box_center_y as f32;

                // rotate back
                final_x = (new_y - self.physical_address_of_box_center_y as f32)
                    * angle_in_radians.sin()
                    + (new_x - self.physical_address_of_box_center_x as f32)
                        * angle_in_radians.cos();
                final_y = (new_y - self.physical_address_of_box_center_y as f32)
                    * angle_in_radians.cos()
                    - (new_x - self.physical_address_of_box_center_x as f32)
                        * angle_in_radians.sin();

                final_x += self.physical_address_of_box_center_x as f32;
                final_y += self.physical_address_of_box_center_y as f32;

                if final_x < 0.0
                    || final_x > (self.logical_x_dimension - 1) as f32
                    || final_y < 0.0
                    || final_y > (self.logical_y_dimension - 1) as f32
                {
                    self.real_values[pixel_counter as usize] = average_edge_value;
                } else {
                    buffer_image.real_values[pixel_counter as usize] =
                        self.return_linear_interpolated_2d(final_x, final_y);
                }

                pixel_counter += 1;
            }

            pixel_counter += i64::from(self.padding_jump_value);
        }

        self.consume(&mut buffer_image);
    }
}

#[cfg(test)]
mod tests {
    use super::Image;

    #[test]
    fn allocation_sets_the_source_padding_and_bounds() {
        let mut image = Image::new();
        image.allocate_2d(8, 6, true);
        assert_eq!(image.padding_jump_value, 2);
        assert_eq!(image.real_memory_allocated, 5 * 6 * 2);
        assert_eq!(image.physical_address_of_box_center_x, 4);
        assert_eq!(image.logical_upper_bound_real_x, 3);
        assert!((image.fourier_voxel_size_x - 0.125).abs() < 1e-7);
    }

    #[test]
    fn complex_view_aliases_the_real_allocation() {
        let mut image = Image::new();
        image.allocate_2d(4, 4, false);
        image.set_complex_values(3, super::Complex::new(1.5, -2.5));
        assert_eq!(image.real_values[6], 1.5);
        assert_eq!(image.real_values[7], -2.5);
        assert_eq!(image.complex_values(3).abs(), f32::hypot(1.5, -2.5));
    }

    #[test]
    fn edge_average_and_central_cross_follow_source_traversal() {
        let mut image = Image::new();
        image.allocate_2d(4, 4, true);
        image.set_to_constant(2.0);
        assert_eq!(image.return_average_of_real_values_on_edges(), 2.0);
        image.set_maximum_value_on_central_cross(1.0);
        assert_eq!(image.return_real_pixel_from_physical_coord(2, 2, 0), 1.0);
        assert_eq!(image.return_real_pixel_from_physical_coord(0, 0, 0), 2.0);
    }
}
