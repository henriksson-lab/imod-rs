//! Owned Rust foundation for `IMOD/librgctf/image.h` and `image_trim.cpp`.
//!
//! `Image` deliberately has separate real and complex stores.  The C++ class
//! aliases those views onto a malloc allocation; callers choose the active
//! representation through `is_in_real_space`, while Rust keeps ownership and
//! element validity explicit.

use rustfft::{FftPlanner, num_complex::Complex32};

use super::ctf::Ctf;
use super::curve::Curve;
use super::globals::GLOBAL_RANDOM_NUMBER_GENERATOR;

#[derive(Clone, Debug)]
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
    pub real_memory_allocated: usize,
    pub padding_jump_value: i32,
    pub insert_into_which_reconstruction: i32,
    pub number_of_real_space_pixels: usize,
    pub ft_normalization_factor: f32,
    pub real_values: Vec<f32>,
    pub complex_values: Vec<Complex32>,
}

/// A mutable two-dimensional view of one real-space `Image` plane.
///
/// This is the safe Rust equivalent of the source's `Image` object whose
/// storage pointer aliases one 3D slice.
pub struct ImageSliceMut<'a> {
    values: &'a mut [f32],
    logical_x_dimension: i32,
    logical_y_dimension: i32,
    padding_jump_value: i32,
}

impl ImageSliceMut<'_> {
    pub fn get(&self, x: i32, y: i32) -> Option<f32> {
        if x < 0 || y < 0 || x >= self.logical_x_dimension || y >= self.logical_y_dimension {
            return None;
        }
        self.values
            .get(
                y as usize * (self.logical_x_dimension + self.padding_jump_value) as usize
                    + x as usize,
            )
            .copied()
    }
    pub fn set(&mut self, x: i32, y: i32, value: f32) -> bool {
        if x < 0 || y < 0 || x >= self.logical_x_dimension || y >= self.logical_y_dimension {
            return false;
        }
        let address =
            y as usize * (self.logical_x_dimension + self.padding_jump_value) as usize + x as usize;
        self.values[address] = value;
        true
    }
}

impl Default for Image {
    fn default() -> Self {
        Self {
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
            fourier_voxel_size_x: 0.,
            fourier_voxel_size_y: 0.,
            fourier_voxel_size_z: 0.,
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
            ft_normalization_factor: 0.,
            real_values: Vec::new(),
            complex_values: Vec::new(),
        }
    }
}

impl Image {
    /// Reset to the source constructor's initial, unallocated state.
    pub fn setup_initial_values(&mut self) {
        *self = Self::default();
    }
    pub fn allocate(&mut self, x: i32, y: i32, z: i32, real_space: bool) -> Result<(), String> {
        if x <= 0 || y <= 0 || z <= 0 {
            return Err(format!("bad image dimensions: {x}, {y}, {z}"));
        }
        if !self.real_values.is_empty()
            && (
                self.logical_x_dimension,
                self.logical_y_dimension,
                self.logical_z_dimension,
            ) == (x, y, z)
        {
            self.is_in_real_space = real_space;
            return Ok(());
        }
        let real_count = usize::try_from(x)
            .ok()
            .and_then(|x| usize::try_from(y).ok()?.checked_mul(x))
            .and_then(|xy| usize::try_from(z).ok()?.checked_mul(xy))
            .ok_or_else(|| "image allocation overflow".to_owned())?;
        self.logical_x_dimension = x;
        self.logical_y_dimension = y;
        self.logical_z_dimension = z;
        self.is_in_real_space = real_space;
        self.padding_jump_value = if x % 2 == 0 { 2 } else { 1 };
        let complex_x = usize::try_from(x / 2 + 1).unwrap();
        let complex_count = complex_x * y as usize * z as usize;
        self.real_values =
            vec![0.; (x as usize + self.padding_jump_value as usize) * y as usize * z as usize];
        self.complex_values = vec![Complex32::new(0., 0.); complex_count];
        self.real_memory_allocated = self.real_values.len();
        self.number_of_real_space_pixels = real_count;
        self.ft_normalization_factor = 1. / (real_count as f32).sqrt();
        self.update_looping_and_addressing();
        Ok(())
    }
    /// The source's two-dimensional `Allocate` overload.
    pub fn allocate_2d(&mut self, x: i32, y: i32, real_space: bool) -> Result<(), String> {
        self.allocate(x, y, 1, real_space)
    }
    /// Safe equivalent of `AllocateAsPointingToSliceIn3D`.
    pub fn slice_2d_mut(&mut self, slice: usize) -> Option<ImageSliceMut<'_>> {
        if !self.is_in_real_space || slice >= self.logical_z_dimension as usize {
            return None;
        }
        let length = (self.logical_x_dimension + self.padding_jump_value) as usize
            * self.logical_y_dimension as usize;
        let start = slice.checked_mul(length)?;
        let values = self.real_values.get_mut(start..start + length)?;
        Some(ImageSliceMut {
            values,
            logical_x_dimension: self.logical_x_dimension,
            logical_y_dimension: self.logical_y_dimension,
            padding_jump_value: self.padding_jump_value,
        })
    }

    pub fn deallocate(&mut self) {
        self.real_values.clear();
        self.complex_values.clear();
        self.real_memory_allocated = 0;
    }
    pub fn return_smallest_logical_dimension(&self) -> i32 {
        if self.logical_z_dimension == 1 {
            self.logical_x_dimension.min(self.logical_y_dimension)
        } else {
            self.logical_x_dimension
                .min(self.logical_y_dimension)
                .min(self.logical_z_dimension)
        }
    }
    pub fn return_volume_in_real_space(&self) -> usize {
        self.number_of_real_space_pixels
    }
    pub fn return_largest_logical_dimension(&self) -> i32 {
        if self.logical_z_dimension == 1 {
            self.logical_x_dimension.max(self.logical_y_dimension)
        } else {
            self.logical_x_dimension
                .max(self.logical_y_dimension)
                .max(self.logical_z_dimension)
        }
    }
    pub fn return_real_1d_address_from_physical_coord(
        &self,
        x: i32,
        y: i32,
        z: i32,
    ) -> Option<usize> {
        if x < 0
            || y < 0
            || z < 0
            || x >= self.logical_x_dimension
            || y >= self.logical_y_dimension
            || z >= self.logical_z_dimension
        {
            None
        } else {
            Some(
                (z as usize * self.logical_y_dimension as usize + y as usize)
                    * (self.logical_x_dimension + self.padding_jump_value) as usize
                    + x as usize,
            )
        }
    }
    pub fn return_real_pixel_from_physical_coord(&self, x: i32, y: i32, z: i32) -> Option<f32> {
        self.return_real_1d_address_from_physical_coord(x, y, z)
            .and_then(|index| self.real_values.get(index).copied())
    }
    pub fn return_fourier_1d_address_from_physical_coord(
        &self,
        x: i32,
        y: i32,
        z: i32,
    ) -> Option<usize> {
        if x < 0
            || y < 0
            || z < 0
            || x > self.physical_upper_bound_complex_x
            || y > self.physical_upper_bound_complex_y
            || z > self.physical_upper_bound_complex_z
        {
            return None;
        }
        Some(
            ((self.physical_upper_bound_complex_x + 1) as usize
                * (self.physical_upper_bound_complex_y + 1) as usize)
                * z as usize
                + (self.physical_upper_bound_complex_x + 1) as usize * y as usize
                + x as usize,
        )
    }
    pub fn return_fourier_1d_address_from_logical_coord(
        &self,
        x: i32,
        y: i32,
        z: i32,
    ) -> Option<usize> {
        if x < self.logical_lower_bound_complex_x
            || x > self.logical_upper_bound_complex_x
            || y < self.logical_lower_bound_complex_y
            || y > self.logical_upper_bound_complex_y
            || z < self.logical_lower_bound_complex_z
            || z > self.logical_upper_bound_complex_z
        {
            return None;
        }
        let (physical_x, physical_y, physical_z) = if x >= 0 {
            (
                x,
                if y >= 0 {
                    y
                } else {
                    self.logical_y_dimension + y
                },
                if z >= 0 {
                    z
                } else {
                    self.logical_z_dimension + z
                },
            )
        } else {
            (
                -x,
                if y > 0 {
                    self.logical_y_dimension - y
                } else {
                    -y
                },
                if z > 0 {
                    self.logical_z_dimension - z
                } else {
                    -z
                },
            )
        };
        self.return_fourier_1d_address_from_physical_coord(physical_x, physical_y, physical_z)
    }
    pub fn return_complex_pixel_from_logical_coord(
        &self,
        x: i32,
        y: i32,
        z: i32,
        outside: Complex32,
    ) -> Complex32 {
        self.return_fourier_1d_address_from_logical_coord(x, y, z)
            .and_then(|address| self.complex_values.get(address).copied())
            .unwrap_or(outside)
    }
    pub fn return_fourier_logical_coord_given_physical_coord_x(&self, index: i32) -> i32 {
        assert!((0..=self.physical_upper_bound_complex_x).contains(&index));
        if index > self.physical_address_of_box_center_x {
            index - self.logical_x_dimension
        } else {
            index
        }
    }
    pub fn return_fourier_logical_coord_given_physical_coord_y(&self, index: i32) -> i32 {
        assert!((0..=self.physical_upper_bound_complex_y).contains(&index));
        if index >= self.physical_index_of_first_negative_frequency_y {
            index - self.logical_y_dimension
        } else {
            index
        }
    }
    pub fn return_fourier_logical_coord_given_physical_coord_z(&self, index: i32) -> i32 {
        assert!((0..=self.physical_upper_bound_complex_z).contains(&index));
        if index >= self.physical_index_of_first_negative_frequency_z {
            index - self.logical_z_dimension
        } else {
            index
        }
    }
    pub fn fourier_component_has_explicit_hermitian_mate(&self, x: i32, y: i32, z: i32) -> bool {
        let mut explicit = x == 0 && !(y == 0 && z == 0);
        if self.logical_y_dimension % 2 == 0 {
            explicit &= y != self.physical_index_of_first_negative_frequency_y - 1;
        }
        if self.logical_z_dimension > 1 && self.logical_z_dimension % 2 == 0 {
            explicit &= z != self.physical_index_of_first_negative_frequency_z - 1;
        }
        explicit
    }
    pub fn fourier_component_is_explicit_hermitian_mate(&self, x: i32, y: i32, z: i32) -> bool {
        x == 0
            && (y >= self.physical_index_of_first_negative_frequency_y
                || z >= self.physical_index_of_first_negative_frequency_z)
    }
    pub fn set_real_pixel_from_physical_coord(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        value: f32,
    ) -> bool {
        let Some(index) = self.return_real_1d_address_from_physical_coord(x, y, z) else {
            return false;
        };
        self.real_values[index] = value;
        true
    }
    pub fn multiply_by_constant(&mut self, value: f32) {
        if self.is_in_real_space {
            for pixel in &mut self.real_values {
                *pixel *= value;
            }
        } else {
            for pixel in &mut self.complex_values {
                *pixel *= value;
            }
        }
    }
    /// Native replacement for the source's FFTW-backed `ForwardFFT`.
    pub fn forward_fft(&mut self, should_scale: bool) {
        assert!(self.is_in_real_space);
        let (width, height, depth) = (
            self.logical_x_dimension as usize,
            self.logical_y_dimension as usize,
            self.logical_z_dimension as usize,
        );
        let mut values = Vec::with_capacity(width * height * depth);
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    values.push(Complex32::new(
                        self.return_real_pixel_from_physical_coord(x, y, z).unwrap(),
                        0.,
                    ));
                }
            }
        }
        let mut planner = FftPlanner::new();
        let fft_x = planner.plan_fft_forward(width);
        for row in values.chunks_exact_mut(width) {
            fft_x.process(row);
        }
        let fft_y = planner.plan_fft_forward(height);
        for z in 0..depth {
            for x in 0..width {
                let mut line = (0..height)
                    .map(|y| values[(z * height + y) * width + x])
                    .collect::<Vec<_>>();
                fft_y.process(&mut line);
                for (y, value) in line.into_iter().enumerate() {
                    values[(z * height + y) * width + x] = value;
                }
            }
        }
        let fft_z = planner.plan_fft_forward(depth);
        for y in 0..height {
            for x in 0..width {
                let mut line = (0..depth)
                    .map(|z| values[(z * height + y) * width + x])
                    .collect::<Vec<_>>();
                fft_z.process(&mut line);
                for (z, value) in line.into_iter().enumerate() {
                    values[(z * height + y) * width + x] = value;
                }
            }
        }
        let scale = if should_scale {
            1. / self.number_of_real_space_pixels as f32
        } else {
            1.
        };
        for z in 0..depth {
            for y in 0..height {
                for x in 0..=width / 2 {
                    let address = self
                        .return_fourier_1d_address_from_physical_coord(x as i32, y as i32, z as i32)
                        .unwrap();
                    self.complex_values[address] = values[(z * height + y) * width + x] * scale;
                }
            }
        }
        self.is_in_real_space = false;
    }
    /// Native replacement for the source's FFTW-backed `BackwardFFT`.
    pub fn backward_fft(&mut self) {
        assert!(!self.is_in_real_space);
        let (width, height, depth) = (
            self.logical_x_dimension as usize,
            self.logical_y_dimension as usize,
            self.logical_z_dimension as usize,
        );
        let mut values = vec![Complex32::new(0., 0.); width * height * depth];
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    values[(z * height + y) * width + x] = if x <= width / 2 {
                        self.complex_values[self
                            .return_fourier_1d_address_from_physical_coord(
                                x as i32, y as i32, z as i32,
                            )
                            .unwrap()]
                    } else {
                        self.complex_values[self
                            .return_fourier_1d_address_from_physical_coord(
                                (width - x) as i32,
                                ((height - y) % height) as i32,
                                ((depth - z) % depth) as i32,
                            )
                            .unwrap()]
                        .conj()
                    };
                }
            }
        }
        let mut planner = FftPlanner::new();
        let fft_x = planner.plan_fft_inverse(width);
        for row in values.chunks_exact_mut(width) {
            fft_x.process(row);
        }
        let fft_y = planner.plan_fft_inverse(height);
        for z in 0..depth {
            for x in 0..width {
                let mut line = (0..height)
                    .map(|y| values[(z * height + y) * width + x])
                    .collect::<Vec<_>>();
                fft_y.process(&mut line);
                for (y, value) in line.into_iter().enumerate() {
                    values[(z * height + y) * width + x] = value;
                }
            }
        }
        let fft_z = planner.plan_fft_inverse(depth);
        for y in 0..height {
            for x in 0..width {
                let mut line = (0..depth)
                    .map(|z| values[(z * height + y) * width + x])
                    .collect::<Vec<_>>();
                fft_z.process(&mut line);
                for (z, value) in line.into_iter().enumerate() {
                    values[(z * height + y) * width + x] = value;
                }
            }
        }
        let scale = 1. / self.number_of_real_space_pixels as f32;
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    self.set_real_pixel_from_physical_coord(
                        x as i32,
                        y as i32,
                        z as i32,
                        values[(z * height + y) * width + x].re * scale,
                    );
                }
            }
        }
        self.is_in_real_space = true;
    }
    pub fn divide_by_constant(&mut self, value: f32) {
        self.multiply_by_constant(1. / value);
    }
    pub fn multiply_add_constant(&mut self, multiply: f32, add: f32) {
        for value in &mut self.real_values {
            *value = *value * multiply + add;
        }
    }
    pub fn add_multiply_constant(&mut self, add: f32, multiply: f32) {
        for value in &mut self.real_values {
            *value = (*value + add) * multiply;
        }
    }
    pub fn add_multiply_add_constant(&mut self, first_add: f32, multiply: f32, second_add: f32) {
        for value in &mut self.real_values {
            *value = (*value + first_add) * multiply + second_add;
        }
    }
    pub fn take_reciprocal_real_values(&mut self, zero_value: f32) {
        assert!(self.is_in_real_space);
        for value in &mut self.real_values {
            *value = if *value == 0. {
                zero_value
            } else {
                1. / *value
            };
        }
    }
    pub fn invert_real_values(&mut self) {
        assert!(self.is_in_real_space);
        for value in &mut self.real_values {
            *value = -*value;
        }
    }
    pub fn square_real_values(&mut self) {
        assert!(self.is_in_real_space);
        for value in &mut self.real_values {
            *value *= *value;
        }
    }
    pub fn exponentiate_real_values(&mut self) {
        assert!(self.is_in_real_space);
        for value in &mut self.real_values {
            *value = value.exp();
        }
    }
    pub fn square_root_real_values(&mut self) {
        assert!(self.is_in_real_space && !self.has_negative_real_value());
        for value in &mut self.real_values {
            *value = value.sqrt();
        }
    }
    pub fn is_constant(&self) -> bool {
        self.real_values
            .first()
            .is_none_or(|first| self.real_values.iter().all(|value| value == first))
    }
    pub fn is_binary(&self) -> bool {
        self.is_in_real_space && self.real_pixels().all(|value| value == 0. || value == 1.)
    }
    pub fn has_nan(&self) -> bool {
        self.real_pixels().any(f32::is_nan)
    }
    pub fn has_negative_real_value(&self) -> bool {
        self.real_pixels().any(|value| value < 0.)
    }
    pub fn add_constant(&mut self, value: f32) {
        assert!(self.is_in_real_space);
        for pixel in &mut self.real_values {
            *pixel += value;
        }
    }
    pub fn set_to_constant(&mut self, value: f32) {
        assert!(self.is_in_real_space);
        for pixel in &mut self.real_values {
            *pixel = value;
        }
    }
    pub fn add_fftw_padding(&mut self) {
        let width = self.logical_x_dimension as usize;
        let stride = width + self.padding_jump_value as usize;
        for z in (0..self.logical_z_dimension as usize).rev() {
            for y in (0..self.logical_y_dimension as usize).rev() {
                let source = (z * self.logical_y_dimension as usize + y) * width;
                let destination = (z * self.logical_y_dimension as usize + y) * stride;
                self.real_values
                    .copy_within(source..source + width, destination);
            }
        }
    }
    pub fn remove_fftw_padding(&mut self) {
        let width = self.logical_x_dimension as usize;
        let stride = width + self.padding_jump_value as usize;
        for z in 0..self.logical_z_dimension as usize {
            for y in 0..self.logical_y_dimension as usize {
                let source = (z * self.logical_y_dimension as usize + y) * stride;
                let destination = (z * self.logical_y_dimension as usize + y) * width;
                self.real_values
                    .copy_within(source..source + width, destination);
            }
        }
    }
    pub fn set_maximum_value(&mut self, maximum: f32) {
        for index in self.real_pixel_indices().collect::<Vec<_>>() {
            self.real_values[index] = self.real_values[index].min(maximum);
        }
    }
    pub fn set_minimum_value(&mut self, minimum: f32) {
        for index in self.real_pixel_indices().collect::<Vec<_>>() {
            self.real_values[index] = self.real_values[index].max(minimum);
        }
    }
    pub fn set_minimum_and_maximum_values(&mut self, minimum: f32, maximum: f32) {
        for index in self.real_pixel_indices().collect::<Vec<_>>() {
            self.real_values[index] = self.real_values[index].clamp(minimum, maximum);
        }
    }
    pub fn normalize_ft(&mut self) {
        self.multiply_by_constant(self.ft_normalization_factor);
    }
    pub fn normalize_ft_and_invert_real_values(&mut self) {
        self.multiply_by_constant(-self.ft_normalization_factor);
    }
    pub fn binarise(&mut self, threshold: f32) {
        assert!(self.is_in_real_space);
        for value in &mut self.real_values {
            *value = if *value >= threshold { 1. } else { 0. };
        }
    }
    pub fn return_maximum_diagonal_radius(&self) -> f32 {
        if self.is_in_real_space {
            ((self.physical_address_of_box_center_x.pow(2)
                + self.physical_address_of_box_center_y.pow(2)
                + self.physical_address_of_box_center_z.pow(2)) as f32)
                .sqrt()
        } else {
            ((self.logical_lower_bound_complex_x as f32 * self.fourier_voxel_size_x).powi(2)
                + (self.logical_lower_bound_complex_y as f32 * self.fourier_voxel_size_y).powi(2)
                + (self.logical_lower_bound_complex_z as f32 * self.fourier_voxel_size_z).powi(2))
            .sqrt()
        }
    }
    pub fn get_min_max(&self) -> Option<(f32, f32)> {
        let mut values = self.real_pixels();
        let first = values.next()?;
        Some(values.fold((first, first), |(minimum, maximum), value| {
            (minimum.min(value), maximum.max(value))
        }))
    }
    pub fn return_average_of_real_values(&self, radius: f32, invert_mask: bool) -> f32 {
        let mut sum = 0.;
        let mut count = 0;
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    let dx = x - self.physical_address_of_box_center_x;
                    let dy = y - self.physical_address_of_box_center_y;
                    let dz = z - self.physical_address_of_box_center_z;
                    let include = radius <= 0.
                        || ((dx * dx + dy * dy + dz * dz) as f32 <= radius * radius) != invert_mask;
                    if include {
                        sum += self.return_real_pixel_from_physical_coord(x, y, z).unwrap();
                        count += 1;
                    }
                }
            }
        }
        if count == 0 { 0. } else { sum / count as f32 }
    }
    pub fn return_average_of_real_values_at_radius(&self, radius: f32) -> f32 {
        let mut sum = 0.;
        let mut count = 0;
        let target = radius * radius;
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    let dx = x - self.physical_address_of_box_center_x;
                    let dy = y - self.physical_address_of_box_center_y;
                    let dz = z - self.physical_address_of_box_center_z;
                    if ((dx * dx + dy * dy + dz * dz) as f32 - target).abs() <= 4. {
                        sum += self.return_real_pixel_from_physical_coord(x, y, z).unwrap();
                        count += 1;
                    }
                }
            }
        }
        if count == 0 { 0. } else { sum / count as f32 }
    }
    pub fn return_maximum_value(
        &self,
        minimum_distance_from_center: f32,
        minimum_distance_from_edge: f32,
    ) -> f32 {
        assert!(self.is_in_real_space);
        let last_x = self.logical_x_dimension as f32 - minimum_distance_from_edge - 1.;
        let last_y = self.logical_y_dimension as f32 - minimum_distance_from_edge - 1.;
        let last_z = self.logical_z_dimension as f32 - minimum_distance_from_edge - 1.;
        let mut result = -f32::MAX;
        for z in 0..self.logical_z_dimension {
            if self.logical_z_dimension > 1
                && (((z - self.physical_address_of_box_center_z).unsigned_abs() as f32)
                    < minimum_distance_from_center
                    || (z as f32) < minimum_distance_from_edge
                    || (z as f32) > last_z)
            {
                continue;
            }
            for y in 0..self.logical_y_dimension {
                if ((y - self.physical_address_of_box_center_y).unsigned_abs() as f32)
                    < minimum_distance_from_center
                    || (y as f32) < minimum_distance_from_edge
                    || (y as f32) > last_y
                {
                    continue;
                }
                for x in 0..self.logical_x_dimension {
                    if ((x - self.physical_address_of_box_center_x).unsigned_abs() as f32)
                        >= minimum_distance_from_center
                        && (x as f32) >= minimum_distance_from_edge
                        && (x as f32) <= last_x
                    {
                        result = result
                            .max(self.return_real_pixel_from_physical_coord(x, y, z).unwrap());
                    }
                }
            }
        }
        result
    }
    pub fn return_minimum_value(
        &self,
        minimum_distance_from_center: f32,
        minimum_distance_from_edge: f32,
    ) -> f32 {
        assert!(self.is_in_real_space);
        let last_x = self.logical_x_dimension as f32 - minimum_distance_from_edge - 1.;
        let last_y = self.logical_y_dimension as f32 - minimum_distance_from_edge - 1.;
        let last_z = self.logical_z_dimension as f32 - minimum_distance_from_edge - 1.;
        let mut result = f32::MAX;
        for z in 0..self.logical_z_dimension {
            if self.logical_z_dimension > 1
                && (((z - self.physical_address_of_box_center_z).unsigned_abs() as f32)
                    < minimum_distance_from_center
                    || (z as f32) < minimum_distance_from_edge
                    || (z as f32) > last_z)
            {
                continue;
            }
            for y in 0..self.logical_y_dimension {
                if ((y - self.physical_address_of_box_center_y).unsigned_abs() as f32)
                    < minimum_distance_from_center
                    || (y as f32) < minimum_distance_from_edge
                    || (y as f32) > last_y
                {
                    continue;
                }
                for x in 0..self.logical_x_dimension {
                    if ((x - self.physical_address_of_box_center_x).unsigned_abs() as f32)
                        >= minimum_distance_from_center
                        && (x as f32) >= minimum_distance_from_edge
                        && (x as f32) <= last_x
                    {
                        result = result
                            .min(self.return_real_pixel_from_physical_coord(x, y, z).unwrap());
                    }
                }
            }
        }
        result
    }
    pub fn return_median_of_real_values(&self) -> Option<f32> {
        let mut values = self.real_pixels().collect::<Vec<_>>();
        if values.is_empty() {
            return None;
        }
        let middle = values.len() / 2;
        values.select_nth_unstable_by(middle, f32::total_cmp);
        Some(values[middle])
    }
    pub fn return_average_of_real_values_on_edges(&self) -> f32 {
        let mut sum = 0.;
        let mut count = 0;
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    if x == 0
                        || y == 0
                        || x == self.logical_x_dimension - 1
                        || y == self.logical_y_dimension - 1
                        || (self.logical_z_dimension > 1
                            && (z == 0 || z == self.logical_z_dimension - 1))
                    {
                        sum += self.return_real_pixel_from_physical_coord(x, y, z).unwrap();
                        count += 1;
                    }
                }
            }
        }
        sum / count as f32
    }
    pub fn circle_mask_with_value(&mut self, radius: f32, mask_value: f32, invert: bool) {
        assert!(self.is_in_real_space);
        let radius_squared = radius * radius;
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    let dx = x - self.physical_address_of_box_center_x;
                    let dy = y - self.physical_address_of_box_center_y;
                    let dz = z - self.physical_address_of_box_center_z;
                    if (((dx * dx + dy * dy + dz * dz) as f32 <= radius_squared) == invert) {
                        self.set_real_pixel_from_physical_coord(x, y, z, mask_value);
                    }
                }
            }
        }
    }
    pub fn circle_mask(&mut self, radius: f32, invert: bool) {
        assert!(self.is_in_real_space && self.object_is_centred_in_box);
        let mask_value = self.return_average_of_real_values_at_radius(radius);
        self.circle_mask_with_value(radius, mask_value, invert);
    }
    pub fn square_mask_with_value(&mut self, dimension: f32, mask_value: f32, invert: bool) {
        assert!(self.is_in_real_space);
        let half = dimension / 2.;
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    let inside = (x - self.physical_address_of_box_center_x).unsigned_abs() as f32
                        <= half
                        && (y - self.physical_address_of_box_center_y).unsigned_abs() as f32
                            <= half
                        && (z - self.physical_address_of_box_center_z).unsigned_abs() as f32
                            <= half;
                    if inside == invert {
                        self.set_real_pixel_from_physical_coord(x, y, z, mask_value);
                    }
                }
            }
        }
    }
    pub fn cosine_mask(
        &mut self,
        radius: f32,
        edge: f32,
        invert: bool,
        force_value: Option<f32>,
    ) -> f32 {
        assert!(edge > 0.);
        let inner = (radius - edge / 2.).max(0.);
        let outer = inner + edge;
        if !self.is_in_real_space {
            for z in 0..=self.physical_upper_bound_complex_z {
                let fz = self.return_fourier_logical_coord_given_physical_coord_z(z) as f32
                    * self.fourier_voxel_size_z;
                for y in 0..=self.physical_upper_bound_complex_y {
                    let fy = self.return_fourier_logical_coord_given_physical_coord_y(y) as f32
                        * self.fourier_voxel_size_y;
                    for x in 0..=self.physical_upper_bound_complex_x {
                        let frequency = ((x as f32 * self.fourier_voxel_size_x).powi(2)
                            + fy.powi(2)
                            + fz.powi(2))
                        .sqrt();
                        let factor = if frequency >= outer {
                            0.
                        } else if frequency <= inner {
                            1.
                        } else {
                            (1. + (std::f32::consts::PI * (frequency - inner) / edge).cos()) / 2.
                        };
                        let factor = if invert { 1. - factor } else { factor };
                        let address = self
                            .return_fourier_1d_address_from_physical_coord(x, y, z)
                            .unwrap();
                        self.complex_values[address] *= factor;
                    }
                }
            }
            return 0.;
        }
        let background = force_value.unwrap_or_else(|| {
            let mut sum = 0.;
            let mut count = 0;
            for z in 0..self.logical_z_dimension {
                for y in 0..self.logical_y_dimension {
                    for x in 0..self.logical_x_dimension {
                        let dx = if self.object_is_centred_in_box {
                            x - self.physical_address_of_box_center_x
                        } else if x >= self.physical_address_of_box_center_x {
                            x - self.logical_x_dimension
                        } else {
                            x
                        };
                        let dy = if self.object_is_centred_in_box {
                            y - self.physical_address_of_box_center_y
                        } else if y >= self.physical_address_of_box_center_y {
                            y - self.logical_y_dimension
                        } else {
                            y
                        };
                        let dz = if self.object_is_centred_in_box {
                            z - self.physical_address_of_box_center_z
                        } else if z >= self.physical_address_of_box_center_z {
                            z - self.logical_z_dimension
                        } else {
                            z
                        };
                        let distance = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                        if distance >= inner && distance <= outer {
                            sum += self.return_real_pixel_from_physical_coord(x, y, z).unwrap();
                            count += 1;
                        }
                    }
                }
            }
            if count == 0 { 0. } else { sum / count as f32 }
        });
        let mut volume = 0.;
        for z in 0..self.logical_z_dimension {
            for y in 0..self.logical_y_dimension {
                for x in 0..self.logical_x_dimension {
                    let dx = if self.object_is_centred_in_box {
                        x - self.physical_address_of_box_center_x
                    } else if x >= self.physical_address_of_box_center_x {
                        x - self.logical_x_dimension
                    } else {
                        x
                    };
                    let dy = if self.object_is_centred_in_box {
                        y - self.physical_address_of_box_center_y
                    } else if y >= self.physical_address_of_box_center_y {
                        y - self.logical_y_dimension
                    } else {
                        y
                    };
                    let dz = if self.object_is_centred_in_box {
                        z - self.physical_address_of_box_center_z
                    } else if z >= self.physical_address_of_box_center_z {
                        z - self.logical_z_dimension
                    } else {
                        z
                    };
                    let distance = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                    let keep = if distance <= inner {
                        1.
                    } else if distance >= outer {
                        0.
                    } else {
                        (1. + (std::f32::consts::PI * (distance - inner) / edge).cos()) / 2.
                    };
                    let keep = if self.object_is_centred_in_box && invert {
                        1. - keep
                    } else {
                        keep
                    };
                    let value = self.return_real_pixel_from_physical_coord(x, y, z).unwrap();
                    self.set_real_pixel_from_physical_coord(
                        x,
                        y,
                        z,
                        value * keep + background * (1. - keep),
                    );
                    volume += keep * keep;
                }
            }
        }
        volume
    }
    pub fn compute_average_and_sigma_of_values_in_spectrum(
        &self,
        minimum_radius: f32,
        maximum_radius: f32,
        cross_half_width: i32,
    ) -> Option<(f32, f32)> {
        assert!(
            self.is_in_real_space
                && self.logical_z_dimension == 1
                && maximum_radius > minimum_radius
        );
        let minimum_squared = minimum_radius.powi(2);
        let maximum_squared = maximum_radius.powi(2);
        let cross_squared = cross_half_width.pow(2);
        let values = (0..self.logical_y_dimension)
            .flat_map(|y| {
                (0..self.logical_x_dimension).filter_map(move |x| {
                    let y_squared = (y - self.physical_address_of_box_center_y).pow(2);
                    let x_squared = (x - self.physical_address_of_box_center_x).pow(2);
                    (x_squared > cross_squared
                        && y_squared > cross_squared
                        && ((x_squared + y_squared) as f32) > minimum_squared
                        && ((x_squared + y_squared) as f32) < maximum_squared)
                        .then(|| self.return_real_pixel_from_physical_coord(x, y, 0).unwrap())
                })
            })
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| {
            let average = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values
                .iter()
                .map(|value| (value - average).powi(2))
                .sum::<f32>()
                / values.len() as f32;
            (average, variance.sqrt())
        })
    }
    pub fn zero_central_pixel(&mut self) {
        assert_eq!(self.logical_z_dimension, 1);
        if self.is_in_real_space {
            self.set_real_pixel_from_physical_coord(
                self.physical_address_of_box_center_x,
                self.physical_address_of_box_center_y,
                0,
                0.,
            );
        } else {
            self.complex_values[0] = Complex32::new(0., 0.);
        }
    }
    pub fn set_maximum_value_on_central_cross(&mut self, maximum: f32) {
        assert!(self.is_in_real_space && self.logical_z_dimension == 1);
        for y in 0..self.logical_y_dimension {
            for x in 0..self.logical_x_dimension {
                if y == self.physical_address_of_box_center_y
                    || x == self.physical_address_of_box_center_x
                {
                    let address = self
                        .return_real_1d_address_from_physical_coord(x, y, 0)
                        .unwrap();
                    self.real_values[address] = self.real_values[address].min(maximum);
                }
            }
        }
    }
    pub fn get_correlation_with_ctf(&self, ctf: &Ctf) -> f32 {
        assert!(self.is_in_real_space && self.logical_z_dimension == 1);
        assert!(ctf.lowest_frequency_for_fitting() > 0.);
        let mut cross_product = 0_f64;
        let mut image_norm = 0_f64;
        let mut ctf_norm = 0_f64;
        let mut number_of_values = 0_usize;
        let lowest = ctf.lowest_frequency_for_fitting().powi(2);
        let highest = ctf.highest_frequency_for_fitting().powi(2);
        for y in 0..self.logical_y_dimension {
            if (self.physical_address_of_box_center_y - 10
                ..=self.physical_address_of_box_center_y + 10)
                .contains(&y)
            {
                continue;
            }
            let fy = (y - self.physical_address_of_box_center_y) as f32
                / self.logical_y_dimension as f32;
            for x in 0..self.physical_address_of_box_center_x - 10 {
                let fx = (x - self.physical_address_of_box_center_x) as f32
                    / self.logical_x_dimension as f32;
                let frequency_squared = fx.powi(2) + fy.powi(2);
                if frequency_squared > lowest && frequency_squared < highest {
                    let value = self.return_real_pixel_from_physical_coord(x, y, 0).unwrap();
                    let ctf_value = ctf.evaluate(frequency_squared, fy.atan2(fx)).abs();
                    number_of_values += 1;
                    cross_product += (value * ctf_value) as f64;
                    image_norm += value.powi(2) as f64;
                    ctf_norm += ctf_value.powi(2) as f64;
                }
            }
        }
        let penalty = if ctf.astigmatism_tolerance() > 0. {
            ctf.astigmatism().powi(2) as f64 * 0.5
                / ctf.astigmatism_tolerance().powi(2) as f64
                / number_of_values as f64
        } else {
            0.
        };
        (cross_product / (image_norm * ctf_norm).sqrt() - penalty) as f32
    }
    pub fn setup_quick_correlation_with_ctf(
        &self,
        ctf: &Ctf,
    ) -> (Vec<usize>, Vec<f32>, Vec<f32>, f64, f64) {
        assert!(self.is_in_real_space && self.logical_z_dimension == 1);
        assert!(ctf.lowest_frequency_for_fitting() > 0.);
        let mut addresses = Vec::new();
        let mut frequencies = Vec::new();
        let mut azimuths = Vec::new();
        let lowest = ctf.lowest_frequency_for_fitting().powi(2);
        let highest = ctf.highest_frequency_for_fitting().powi(2);
        for y in 0..self.logical_y_dimension {
            if (self.physical_address_of_box_center_y - 10
                ..=self.physical_address_of_box_center_y + 10)
                .contains(&y)
            {
                continue;
            }
            let fy = (y - self.physical_address_of_box_center_y) as f32
                / self.logical_y_dimension as f32;
            for x in 0..self.physical_address_of_box_center_x - 10 {
                let fx = (x - self.physical_address_of_box_center_x) as f32
                    / self.logical_x_dimension as f32;
                let frequency_squared = fx.powi(2) + fy.powi(2);
                if frequency_squared > lowest && frequency_squared < highest {
                    addresses.push(
                        self.return_real_1d_address_from_physical_coord(x, y, 0)
                            .unwrap(),
                    );
                    frequencies.push(frequency_squared);
                    azimuths.push(fy.atan2(fx));
                }
            }
        }
        let mean = addresses
            .iter()
            .map(|&address| self.real_values[address] as f64)
            .sum::<f64>()
            / addresses.len() as f64;
        let norm = addresses
            .iter()
            .map(|&address| (self.real_values[address] as f64 - mean).powi(2))
            .sum();
        (addresses, frequencies, azimuths, norm, mean)
    }
    pub fn quick_correlation_with_ctf(
        &self,
        ctf: &Ctf,
        addresses: &[usize],
        frequency_squared: &[f32],
        azimuths: &[f32],
        image_norm: f64,
        image_mean: f64,
    ) -> f32 {
        assert_eq!(addresses.len(), frequency_squared.len());
        assert_eq!(addresses.len(), azimuths.len());
        let mut cross_product = 0_f64;
        let mut ctf_norm = 0_f64;
        let mut ctf_sum = 0_f64;
        for ((&address, &frequency), &azimuth) in
            addresses.iter().zip(frequency_squared).zip(azimuths)
        {
            let value = -ctf
                .phase_shift_given_squared_spatial_frequency_and_azimuth(frequency, azimuth)
                .sin();
            cross_product += self.real_values[address] as f64 * value as f64;
            ctf_norm += value.powi(2) as f64;
            ctf_sum += value as f64;
        }
        let penalty = if ctf.astigmatism_tolerance() > 0. {
            ctf.astigmatism().powi(2) as f64 * 0.5
                / ctf.astigmatism_tolerance().powi(2) as f64
                / addresses.len() as f64
        } else {
            0.
        };
        ((cross_product - image_mean * ctf_sum)
            / (image_norm * (ctf_norm - ctf_sum.powi(2) / addresses.len() as f64)).sqrt()
            - penalty) as f32
    }
    pub fn compute_1d_rotational_average(
        &self,
        average: &mut Curve,
        number_of_values: &mut Curve,
        fractional_radius_in_real_space: bool,
    ) {
        assert_eq!(average.number_of_points, number_of_values.number_of_points);
        average.zero_y_data();
        number_of_values.zero_y_data();
        if self.is_in_real_space {
            for z in 0..self.logical_z_dimension {
                for y in 0..self.logical_y_dimension {
                    for x in 0..self.logical_x_dimension {
                        let radius = if fractional_radius_in_real_space {
                            (((x - self.physical_address_of_box_center_x) as f32
                                * self.fourier_voxel_size_x)
                                .powi(2)
                                + ((y - self.physical_address_of_box_center_y) as f32
                                    * self.fourier_voxel_size_y)
                                    .powi(2)
                                + ((z - self.physical_address_of_box_center_z) as f32
                                    * self.fourier_voxel_size_z)
                                    .powi(2))
                            .sqrt()
                        } else {
                            (((x - self.physical_address_of_box_center_x).pow(2)
                                + (y - self.physical_address_of_box_center_y).pow(2)
                                + (z - self.physical_address_of_box_center_z).pow(2))
                                as f32)
                                .sqrt()
                        };
                        average.add_value_at_x_using_linear_interpolation(
                            radius,
                            self.return_real_pixel_from_physical_coord(x, y, z).unwrap(),
                            true,
                        );
                        number_of_values
                            .add_value_at_x_using_linear_interpolation(radius, 1., true);
                    }
                }
            }
        } else {
            for z in 0..=self.physical_upper_bound_complex_z {
                let z_squared = (self.return_fourier_logical_coord_given_physical_coord_z(z)
                    as f32
                    * self.fourier_voxel_size_z)
                    .powi(2);
                for y in 0..=self.physical_upper_bound_complex_y {
                    let y_squared = (self.return_fourier_logical_coord_given_physical_coord_y(y)
                        as f32
                        * self.fourier_voxel_size_y)
                        .powi(2);
                    for x in 0..self.physical_upper_bound_complex_x {
                        if self.fourier_component_is_explicit_hermitian_mate(x, y, z) {
                            continue;
                        }
                        let radius = ((x as f32 * self.fourier_voxel_size_x).powi(2)
                            + y_squared
                            + z_squared)
                            .sqrt();
                        average.add_value_at_x_using_linear_interpolation(
                            radius,
                            self.complex_values[self
                                .return_fourier_1d_address_from_physical_coord(x, y, z)
                                .unwrap()]
                            .norm(),
                            true,
                        );
                        number_of_values
                            .add_value_at_x_using_linear_interpolation(radius, 1., true);
                    }
                }
            }
        }
        for (sum, count) in average.data_y.iter_mut().zip(&number_of_values.data_y) {
            if *count != 0. {
                *sum /= count;
            }
        }
    }
    pub fn compute_1d_power_spectrum_curve(
        &self,
        average_power: &mut Curve,
        number_of_values: &mut Curve,
    ) {
        assert!(!self.is_in_real_space && average_power.number_of_points > 0);
        assert!(average_power.data_x[0] == 0. && *average_power.data_x.last().unwrap() >= 0.5);
        assert_eq!(
            average_power.number_of_points,
            number_of_values.number_of_points
        );
        assert_eq!(
            average_power.data_x.first(),
            number_of_values.data_x.first()
        );
        assert_eq!(average_power.data_x.last(), number_of_values.data_x.last());
        average_power.zero_y_data();
        number_of_values.zero_y_data();
        for z in 0..=self.physical_upper_bound_complex_z {
            let z_squared = (self.return_fourier_logical_coord_given_physical_coord_z(z) as f32
                * self.fourier_voxel_size_z)
                .powi(2);
            for y in 0..=self.physical_upper_bound_complex_y {
                let y_squared = (self.return_fourier_logical_coord_given_physical_coord_y(y)
                    as f32
                    * self.fourier_voxel_size_y)
                    .powi(2);
                for x in 0..=self.physical_upper_bound_complex_x {
                    if self.fourier_component_is_explicit_hermitian_mate(x, y, z) {
                        continue;
                    }
                    let frequency =
                        ((x as f32 * self.fourier_voxel_size_x).powi(2) + y_squared + z_squared)
                            .sqrt();
                    let value = self.complex_values[self
                        .return_fourier_1d_address_from_physical_coord(x, y, z)
                        .unwrap()]
                    .norm_sqr();
                    average_power.add_value_at_x_using_linear_interpolation(frequency, value, true);
                    number_of_values.add_value_at_x_using_linear_interpolation(frequency, 1., true);
                }
            }
        }
        for (sum, count) in average_power
            .data_y
            .iter_mut()
            .zip(&number_of_values.data_y)
        {
            *sum = if *count > 0. { *sum / *count } else { 0. };
        }
    }
    pub fn compute_amplitude_spectrum_full_2d(&self, amplitude: &mut Self) {
        assert!(!self.is_in_real_space && self.has_same_dimensions_as(amplitude));
        assert_eq!(self.logical_z_dimension, 1);
        for y in 0..amplitude.logical_y_dimension {
            for x in 0..amplitude.logical_x_dimension {
                let source = self
                    .return_fourier_1d_address_from_logical_coord(
                        x - amplitude.physical_address_of_box_center_x,
                        y - amplitude.physical_address_of_box_center_y,
                        0,
                    )
                    .unwrap();
                amplitude.set_real_pixel_from_physical_coord(
                    x,
                    y,
                    0,
                    self.complex_values[source].norm(),
                );
            }
        }
        amplitude.is_in_real_space = true;
        amplitude.object_is_centred_in_box = true;
    }
    pub fn spectrum_box_convolution(&self, output: &mut Self, box_size: i32, minimum_radius: f32) {
        assert!(box_size > 0 && box_size % 2 == 1);
        assert_eq!(self.logical_z_dimension, 1);
        assert!(self.has_same_dimensions_as(output));
        let half = box_size / 2;
        let first_x = self.physical_address_of_box_center_x - 1;
        let last_x = self.physical_address_of_box_center_x + 1;
        let first_y = self.physical_address_of_box_center_y - 1;
        let last_y = self.physical_address_of_box_center_y + 1;
        let mut x_ranges = Vec::with_capacity(self.logical_x_dimension as usize);
        for x in 0..self.logical_x_dimension {
            let mut first = x - half;
            let mut last = x + half;
            let mut second = None;
            if first < 0 {
                second = Some((
                    first + self.logical_x_dimension,
                    self.logical_x_dimension - 1,
                ));
                first = 0;
            } else if last >= self.logical_x_dimension {
                second = Some((0, last - self.logical_x_dimension));
                last = self.logical_x_dimension - 1;
            } else if first <= last_x && last >= first_x {
                if first >= first_x {
                    first = last_x + 1;
                } else if last <= last_x {
                    last = first_x - 1;
                } else {
                    second = Some((last_x + 1, last));
                    last = first_x - 1;
                }
            }
            x_ranges.push((first, last, second));
        }
        let mut line_sums =
            vec![0.; (self.logical_x_dimension * self.logical_y_dimension) as usize];
        let mut counts = vec![0_i32; self.logical_x_dimension as usize];
        for (x, (first, last, second)) in x_ranges.iter().enumerate() {
            counts[x] = last + 1 - first + second.map_or(0, |(start, end)| end + 1 - start);
        }
        for y in 0..self.logical_y_dimension {
            for (x, (first, last, second)) in x_ranges.iter().enumerate() {
                let mut sum = 0.;
                for sample_x in *first..=*last {
                    sum += self
                        .return_real_pixel_from_physical_coord(sample_x, y, 0)
                        .unwrap();
                }
                if let Some((start, end)) = second {
                    for sample_x in *start..=*end {
                        sum += self
                            .return_real_pixel_from_physical_coord(sample_x, y, 0)
                            .unwrap();
                    }
                }
                line_sums[x + y as usize * self.logical_x_dimension as usize] = sum;
            }
        }
        for y in 0..self.logical_y_dimension {
            for x in 0..self.logical_x_dimension {
                let radius_squared = (x - self.physical_address_of_box_center_x).pow(2)
                    + (y - self.physical_address_of_box_center_y).pow(2);
                let value = if (radius_squared as f32) <= minimum_radius.powi(2) {
                    self.return_real_pixel_from_physical_coord(x, y, 0).unwrap()
                } else {
                    let mut sum = 0.;
                    let mut count = 0;
                    for offset in -half..=half {
                        let sample_y = (y + offset).rem_euclid(self.logical_y_dimension);
                        if (first_y..=last_y).contains(&sample_y) {
                            continue;
                        }
                        sum += line_sums
                            [x as usize + sample_y as usize * self.logical_x_dimension as usize];
                        count += counts[x as usize];
                    }
                    if count == 0 {
                        self.return_real_pixel_from_physical_coord(x, y, 0).unwrap()
                    } else {
                        sum / count as f32
                    }
                };
                output.set_real_pixel_from_physical_coord(x, y, 0, value);
            }
        }
    }
    /// Copy this centered two-dimensional real-space image into the center of
    /// a larger real-space image, filling its surrounding pixels with
    /// `padding`.  This is the direct owned-buffer counterpart of
    /// `Image::ClipIntoLargerRealSpace2D`.
    pub fn clip_into_larger_real_space_2d(&self, other: &mut Self, padding: f32) {
        assert!(
            !self.real_values.is_empty() && !other.real_values.is_empty(),
            "image memory not allocated"
        );
        assert!(
            self.is_in_real_space && self.object_is_centred_in_box && self.logical_z_dimension == 1,
            "source must be a centered 2D real-space image"
        );
        assert!(
            self.logical_x_dimension <= other.logical_x_dimension
                && self.logical_y_dimension <= other.logical_y_dimension,
            "source must not exceed destination dimensions"
        );

        other.is_in_real_space = self.is_in_real_space;
        other.object_is_centred_in_box = self.object_is_centred_in_box;
        let lower_x =
            other.physical_address_of_box_center_x - self.physical_address_of_box_center_x;
        let lower_y =
            other.physical_address_of_box_center_y - self.physical_address_of_box_center_y;
        let upper_x = lower_x + self.logical_x_dimension - 1;
        let upper_y = lower_y + self.logical_y_dimension - 1;

        for destination_y in 0..other.logical_y_dimension {
            for destination_x in 0..other.logical_x_dimension {
                let value = if destination_x < lower_x
                    || destination_x > upper_x
                    || destination_y < lower_y
                    || destination_y > upper_y
                {
                    padding
                } else {
                    self.return_real_pixel_from_physical_coord(
                        destination_x - lower_x,
                        destination_y - lower_y,
                        0,
                    )
                    .unwrap()
                };
                other.set_real_pixel_from_physical_coord(destination_x, destination_y, 0, value);
            }
        }
    }
    /// Copy a centered real-space image into `other`, optionally centering the
    /// destination at a physical source coordinate.  Pixels outside this
    /// image receive `padding`.
    pub fn clip_into_real_space(
        &self,
        other: &mut Self,
        padding: f32,
        center_x: i32,
        center_y: i32,
        center_z: i32,
    ) {
        assert!(
            !self.real_values.is_empty() && !other.real_values.is_empty(),
            "image memory not allocated"
        );
        assert!(
            self.is_in_real_space && self.object_is_centred_in_box,
            "source must be a centered real-space image"
        );
        other.is_in_real_space = true;
        other.object_is_centred_in_box = self.object_is_centred_in_box;
        for destination_z in 0..other.logical_z_dimension {
            let source_z = self.physical_address_of_box_center_z + center_z + destination_z
                - other.physical_address_of_box_center_z;
            for destination_y in 0..other.logical_y_dimension {
                let source_y = self.physical_address_of_box_center_y + center_y + destination_y
                    - other.physical_address_of_box_center_y;
                for destination_x in 0..other.logical_x_dimension {
                    let source_x = self.physical_address_of_box_center_x + center_x + destination_x
                        - other.physical_address_of_box_center_x;
                    let value = self
                        .return_real_pixel_from_physical_coord(source_x, source_y, source_z)
                        .unwrap_or(padding);
                    other.set_real_pixel_from_physical_coord(
                        destination_x,
                        destination_y,
                        destination_z,
                        value,
                    );
                }
            }
        }
    }
    /// Source `ClipInto`, for both real-space and Fourier images.
    pub fn clip_into(
        &self,
        other: &mut Self,
        padding: f32,
        fill_with_noise: bool,
        noise_sigma: f32,
        center_x: i32,
        center_y: i32,
        center_z: i32,
    ) {
        assert!(
            !self.real_values.is_empty() && !other.real_values.is_empty(),
            "image memory not allocated"
        );
        assert!(
            !self.is_in_real_space || !fill_with_noise,
            "noise fill is only valid in Fourier space"
        );
        assert!(
            self.is_in_real_space || (center_x == 0 && center_y == 0 && center_z == 0),
            "off-center clipping is not valid in Fourier space"
        );
        other.is_in_real_space = self.is_in_real_space;
        other.object_is_centred_in_box = self.object_is_centred_in_box;
        if self.is_in_real_space {
            self.clip_into_real_space(other, padding, center_x, center_y, center_z);
            return;
        }
        let mut generator = if fill_with_noise {
            Some(
                GLOBAL_RANDOM_NUMBER_GENERATOR
                    .lock()
                    .expect("random generator lock poisoned"),
            )
        } else {
            None
        };
        for physical_z in 0..=other.physical_upper_bound_complex_z {
            let logical_z = other.return_fourier_logical_coord_given_physical_coord_z(physical_z);
            for physical_y in 0..=other.physical_upper_bound_complex_y {
                let logical_y =
                    other.return_fourier_logical_coord_given_physical_coord_y(physical_y);
                for physical_x in 0..=other.physical_upper_bound_complex_x {
                    let address = other
                        .return_fourier_1d_address_from_physical_coord(
                            physical_x, physical_y, physical_z,
                        )
                        .unwrap();
                    other.complex_values[address] = if let Some(random) = generator.as_deref_mut() {
                        if physical_x >= self.logical_lower_bound_complex_x
                            && physical_x <= self.logical_upper_bound_complex_x
                            && logical_y >= self.logical_lower_bound_complex_y
                            && logical_y <= self.logical_upper_bound_complex_y
                            && logical_z >= self.logical_lower_bound_complex_z
                            && logical_z <= self.logical_upper_bound_complex_z
                        {
                            self.return_complex_pixel_from_logical_coord(
                                physical_x,
                                logical_y,
                                logical_z,
                                Complex32::new(padding, 0.),
                            )
                        } else {
                            Complex32::new(
                                random.normal_random() * noise_sigma,
                                random.normal_random() * noise_sigma,
                            )
                        }
                    } else {
                        self.return_complex_pixel_from_logical_coord(
                            physical_x,
                            logical_y,
                            logical_z,
                            Complex32::new(padding, 0.),
                        )
                    };
                }
            }
        }
        if self.logical_y_dimension < other.logical_y_dimension
            || self.logical_z_dimension < other.logical_z_dimension
        {
            if self.logical_z_dimension == 1 {
                let y = self.physical_index_of_first_negative_frequency_y;
                for x in 0..=self.physical_upper_bound_complex_x {
                    let source = self
                        .return_fourier_1d_address_from_physical_coord(x, y, 0)
                        .unwrap();
                    let destination = other
                        .return_fourier_1d_address_from_physical_coord(x, y, 0)
                        .unwrap();
                    other.complex_values[destination] = self.complex_values[source];
                }
            } else {
                for logical_z in
                    self.logical_lower_bound_complex_z..=self.logical_upper_bound_complex_z
                {
                    let source_y = self.logical_lower_bound_complex_y;
                    let destination_y = self.physical_index_of_first_negative_frequency_y;
                    for x in 0..=self.physical_upper_bound_complex_x {
                        let source = self
                            .return_fourier_1d_address_from_logical_coord(x, source_y, logical_z)
                            .unwrap();
                        let destination = other
                            .return_fourier_1d_address_from_logical_coord(
                                x,
                                destination_y,
                                logical_z,
                            )
                            .unwrap();
                        other.complex_values[destination] = self.complex_values[source];
                    }
                }
                let z = self.physical_index_of_first_negative_frequency_z;
                let mirrored_z = other.logical_z_dimension - z;
                for y in 1..=self.physical_index_of_first_negative_frequency_y {
                    for x in 0..=self.physical_upper_bound_complex_x {
                        let destination = other
                            .return_fourier_1d_address_from_physical_coord(x, y, z)
                            .unwrap();
                        let source = other
                            .return_fourier_1d_address_from_physical_coord(x, y, mirrored_z)
                            .unwrap();
                        other.complex_values[destination] = other.complex_values[source];
                    }
                }
                for y in other.logical_y_dimension
                    - self.physical_index_of_first_negative_frequency_y
                    ..other.logical_y_dimension
                {
                    for x in 0..=self.physical_upper_bound_complex_x {
                        let destination = other
                            .return_fourier_1d_address_from_physical_coord(x, y, z)
                            .unwrap();
                        let source = other
                            .return_fourier_1d_address_from_physical_coord(x, y, mirrored_z)
                            .unwrap();
                        other.complex_values[destination] = other.complex_values[source];
                    }
                }
                for x in 0..=self.physical_upper_bound_complex_x {
                    let destination = other
                        .return_fourier_1d_address_from_physical_coord(x, 0, z)
                        .unwrap();
                    let source = other
                        .return_fourier_1d_address_from_physical_coord(x, 0, mirrored_z)
                        .unwrap();
                    other.complex_values[destination] = other.complex_values[source];
                }
            }
        }
    }
    /// Resize a real-space image by centered clipping/padding.
    pub fn resize_real_space(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        padding: f32,
    ) -> Result<(), String> {
        assert!(
            self.is_in_real_space,
            "Fourier resize is not this operation"
        );
        if (
            self.logical_x_dimension,
            self.logical_y_dimension,
            self.logical_z_dimension,
        ) == (x, y, z)
        {
            return Ok(());
        }
        let mut resized = Self::default();
        resized.allocate(x, y, z, true)?;
        self.clip_into_real_space(&mut resized, padding, 0, 0, 0);
        *self = resized;
        Ok(())
    }
    pub fn copy_from(&mut self, other: &Self) {
        *self = other.clone();
    }
    pub fn copy_looping_and_addressing_from(&mut self, other: &Self) {
        self.object_is_centred_in_box = other.object_is_centred_in_box;
        self.logical_x_dimension = other.logical_x_dimension;
        self.logical_y_dimension = other.logical_y_dimension;
        self.logical_z_dimension = other.logical_z_dimension;
        self.physical_upper_bound_complex_x = other.physical_upper_bound_complex_x;
        self.physical_upper_bound_complex_y = other.physical_upper_bound_complex_y;
        self.physical_upper_bound_complex_z = other.physical_upper_bound_complex_z;
        self.physical_address_of_box_center_x = other.physical_address_of_box_center_x;
        self.physical_address_of_box_center_y = other.physical_address_of_box_center_y;
        self.physical_address_of_box_center_z = other.physical_address_of_box_center_z;
        self.physical_index_of_first_negative_frequency_y =
            other.physical_index_of_first_negative_frequency_y;
        self.physical_index_of_first_negative_frequency_z =
            other.physical_index_of_first_negative_frequency_z;
        self.fourier_voxel_size_x = other.fourier_voxel_size_x;
        self.fourier_voxel_size_y = other.fourier_voxel_size_y;
        self.fourier_voxel_size_z = other.fourier_voxel_size_z;
        self.logical_upper_bound_complex_x = other.logical_upper_bound_complex_x;
        self.logical_upper_bound_complex_y = other.logical_upper_bound_complex_y;
        self.logical_upper_bound_complex_z = other.logical_upper_bound_complex_z;
        self.logical_lower_bound_complex_x = other.logical_lower_bound_complex_x;
        self.logical_lower_bound_complex_y = other.logical_lower_bound_complex_y;
        self.logical_lower_bound_complex_z = other.logical_lower_bound_complex_z;
        self.logical_upper_bound_real_x = other.logical_upper_bound_real_x;
        self.logical_upper_bound_real_y = other.logical_upper_bound_real_y;
        self.logical_upper_bound_real_z = other.logical_upper_bound_real_z;
        self.logical_lower_bound_real_x = other.logical_lower_bound_real_x;
        self.logical_lower_bound_real_y = other.logical_lower_bound_real_y;
        self.logical_lower_bound_real_z = other.logical_lower_bound_real_z;
        self.padding_jump_value = other.padding_jump_value;
    }
    /// Move all owned image state from `other`, leaving it empty.
    pub fn consume(&mut self, other: &mut Self) {
        *self = std::mem::take(other);
    }
    pub fn get_real_value_by_linear_interpolation_no_bounds_check_image(
        &self,
        x: f32,
        y: f32,
    ) -> f32 {
        assert!(self.is_in_real_space && self.logical_z_dimension == 1);
        let xi = x as i32;
        let yi = y as i32;
        assert!(
            xi >= 0
                && yi >= 0
                && xi + 1 < self.logical_x_dimension
                && yi + 1 < self.logical_y_dimension
        );
        let xd = x - xi as f32;
        let yd = y - yi as f32;
        let a = self
            .return_real_pixel_from_physical_coord(xi, yi, 0)
            .unwrap();
        let b = self
            .return_real_pixel_from_physical_coord(xi + 1, yi, 0)
            .unwrap();
        let c = self
            .return_real_pixel_from_physical_coord(xi, yi + 1, 0)
            .unwrap();
        let d = self
            .return_real_pixel_from_physical_coord(xi + 1, yi + 1, 0)
            .unwrap();
        (1. - xd) * (1. - yd) * a + xd * (1. - yd) * b + (1. - xd) * yd * c + xd * yd * d
    }
    /// Source `ReturnLinearInterpolated2D`: return zero beyond image edges and
    /// use clamped neighbors at the final row or column.
    pub fn return_linear_interpolated_2d(&self, x: f32, y: f32) -> f32 {
        assert!(self.is_in_real_space && self.logical_z_dimension == 1);
        if x < 0.
            || y < 0.
            || x > (self.logical_x_dimension - 1) as f32
            || y > (self.logical_y_dimension - 1) as f32
        {
            return 0.;
        }
        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let x1 = (x0 + 1).min(self.logical_x_dimension - 1);
        let y1 = (y0 + 1).min(self.logical_y_dimension - 1);
        let mut sum = 0.;
        for sample_y in y0..=y1 {
            let weight_y = 1. - (y - sample_y as f32).abs();
            for sample_x in x0..=x1 {
                let weight_x = 1. - (x - sample_x as f32).abs();
                sum += self
                    .return_real_pixel_from_physical_coord(sample_x, sample_y, 0)
                    .unwrap()
                    * weight_x
                    * weight_y;
            }
        }
        sum
    }
    /// Resample a two-dimensional real-space image after anisotropic
    /// magnification along a rotated axis.
    pub fn correct_magnification_distortion(
        &mut self,
        angle_degrees: f32,
        major_axis: f32,
        minor_axis: f32,
    ) {
        assert!(
            self.is_in_real_space && self.logical_z_dimension == 1,
            "only 2D real-space images are supported"
        );
        assert!(major_axis != 0. && minor_axis != 0.);
        let angle = angle_degrees.to_radians();
        let sin_minus = (-angle).sin();
        let cos_minus = (-angle).cos();
        let sin = angle.sin();
        let cos = angle.cos();
        let edge_value = self.return_average_of_real_values_on_edges();
        let mut corrected = Self::default();
        corrected
            .allocate(
                self.logical_x_dimension,
                self.logical_y_dimension,
                self.logical_z_dimension,
                true,
            )
            .unwrap();
        for y in 0..self.logical_y_dimension {
            for x in 0..self.logical_x_dimension {
                let mut transformed_x = (y - self.physical_address_of_box_center_y) as f32
                    * sin_minus
                    + (x - self.physical_address_of_box_center_x) as f32 * cos_minus;
                let mut transformed_y = (y - self.physical_address_of_box_center_y) as f32
                    * cos_minus
                    - (x - self.physical_address_of_box_center_x) as f32 * sin_minus;
                transformed_x /= major_axis;
                transformed_y /= minor_axis;
                transformed_x += self.physical_address_of_box_center_x as f32;
                transformed_y += self.physical_address_of_box_center_y as f32;
                let final_x = (transformed_y - self.physical_address_of_box_center_y as f32) * sin
                    + (transformed_x - self.physical_address_of_box_center_x as f32) * cos
                    + self.physical_address_of_box_center_x as f32;
                let final_y = (transformed_y - self.physical_address_of_box_center_y as f32) * cos
                    - (transformed_x - self.physical_address_of_box_center_x as f32) * sin
                    + self.physical_address_of_box_center_y as f32;
                let value = self.return_linear_interpolated_2d(final_x, final_y);
                corrected.set_real_pixel_from_physical_coord(
                    x,
                    y,
                    0,
                    if final_x < 0.
                        || final_x > (self.logical_x_dimension - 1) as f32
                        || final_y < 0.
                        || final_y > (self.logical_y_dimension - 1) as f32
                    {
                        edge_value
                    } else {
                        value
                    },
                );
            }
        }
        *self = corrected;
    }
    pub fn has_same_dimensions_as(&self, other: &Self) -> bool {
        (
            self.logical_x_dimension,
            self.logical_y_dimension,
            self.logical_z_dimension,
        ) == (
            other.logical_x_dimension,
            other.logical_y_dimension,
            other.logical_z_dimension,
        )
    }
    pub fn multiply_pixel_wise(&mut self, other: &Self) {
        assert!(
            self.has_same_dimensions_as(other) && self.is_in_real_space == other.is_in_real_space
        );
        if self.is_in_real_space {
            for (left, right) in self.real_values.iter_mut().zip(&other.real_values) {
                *left *= right;
            }
        } else {
            for (left, right) in self.complex_values.iter_mut().zip(&other.complex_values) {
                *left *= right;
            }
        }
    }
    pub fn apply_mirror_along_y(&mut self) {
        assert!(self.is_in_real_space && self.logical_z_dimension == 1);
        for y in 1..self.physical_address_of_box_center_y {
            let mirror_y = 2 * (self.physical_address_of_box_center_y - y) + y;
            for x in 0..self.logical_x_dimension {
                let first = self
                    .return_real_1d_address_from_physical_coord(x, y, 0)
                    .unwrap();
                let second = self
                    .return_real_1d_address_from_physical_coord(x, mirror_y, 0)
                    .unwrap();
                self.real_values.swap(first, second);
            }
        }
        let average = (0..self.logical_x_dimension)
            .map(|x| self.return_real_pixel_from_physical_coord(x, 0, 0).unwrap())
            .sum::<f32>()
            / self.logical_x_dimension as f32;
        for x in 0..self.logical_x_dimension {
            self.set_real_pixel_from_physical_coord(x, 0, 0, average);
        }
    }
    pub fn add_image(&mut self, other: &Self) {
        assert_eq!(self.real_values.len(), other.real_values.len());
        for (left, right) in self.real_values.iter_mut().zip(&other.real_values) {
            *left += right;
        }
    }
    pub fn subtract_image(&mut self, other: &Self) {
        assert!(self.has_same_dimensions_as(other));
        for (left, right) in self.real_values.iter_mut().zip(&other.real_values) {
            *left -= right;
        }
    }
    pub fn subtract_squared_image(&mut self, other: &Self) {
        assert_eq!(self.real_values.len(), other.real_values.len());
        for (left, right) in self.real_values.iter_mut().zip(&other.real_values) {
            *left -= right.powi(2);
        }
    }
    pub fn update_looping_and_addressing(&mut self) {
        let (x, y, z) = (
            self.logical_x_dimension,
            self.logical_y_dimension,
            self.logical_z_dimension,
        );
        self.physical_upper_bound_complex_x = x / 2;
        self.physical_upper_bound_complex_y = y - 1;
        self.physical_upper_bound_complex_z = z - 1;
        self.update_physical_address_of_box_center();
        self.physical_index_of_first_negative_frequency_y = y / 2 + y % 2;
        self.physical_index_of_first_negative_frequency_z = z / 2 + z % 2;
        self.fourier_voxel_size_x = 1. / x as f32;
        self.fourier_voxel_size_y = 1. / y as f32;
        self.fourier_voxel_size_z = 1. / z as f32;
        self.logical_lower_bound_complex_x = -x / 2;
        self.logical_upper_bound_complex_x = x / 2;
        self.logical_lower_bound_real_x = -x / 2;
        self.logical_upper_bound_real_x = if x % 2 == 0 { x / 2 - 1 } else { x / 2 };
        for (size, lower_complex, upper_complex, lower_real, upper_real) in [
            (
                y,
                &mut self.logical_lower_bound_complex_y,
                &mut self.logical_upper_bound_complex_y,
                &mut self.logical_lower_bound_real_y,
                &mut self.logical_upper_bound_real_y,
            ),
            (
                z,
                &mut self.logical_lower_bound_complex_z,
                &mut self.logical_upper_bound_complex_z,
                &mut self.logical_lower_bound_real_z,
                &mut self.logical_upper_bound_real_z,
            ),
        ] {
            *lower_complex = -size / 2;
            *upper_complex = if size % 2 == 0 {
                size / 2 - 1
            } else {
                size / 2
            };
            *lower_real = *lower_complex;
            *upper_real = *upper_complex;
        }
    }
    /// Source `SetLogicalDimensions`; callers update dependent addressing
    /// separately, as in the C++ implementation.
    pub fn set_logical_dimensions(&mut self, x: i32, y: i32, z: i32) {
        self.logical_x_dimension = x;
        self.logical_y_dimension = y;
        self.logical_z_dimension = z;
    }
    pub fn update_physical_address_of_box_center(&mut self) {
        self.physical_address_of_box_center_x = self.logical_x_dimension / 2;
        self.physical_address_of_box_center_y = self.logical_y_dimension / 2;
        self.physical_address_of_box_center_z = self.logical_z_dimension / 2;
    }
    fn real_pixel_indices(&self) -> impl Iterator<Item = usize> + '_ {
        let width = self.logical_x_dimension as usize;
        let stride = width + self.padding_jump_value as usize;
        (0..self.logical_z_dimension as usize).flat_map(move |z| {
            (0..self.logical_y_dimension as usize).flat_map(move |y| {
                (0..width).map(move |x| (z * self.logical_y_dimension as usize + y) * stride + x)
            })
        })
    }
    fn real_pixels(&self) -> impl Iterator<Item = f32> + '_ {
        self.real_pixel_indices()
            .map(|index| self.real_values[index])
    }
}

#[cfg(test)]
mod tests {
    use super::{Complex32, Ctf, Curve, Image};
    #[test]
    fn allocation_uses_owned_padded_rows_and_source_bounds() {
        let mut image = Image::default();
        image.allocate(4, 3, 1, true).unwrap();
        assert_eq!(image.real_values.len(), 18);
        assert_eq!(
            image.return_real_1d_address_from_physical_coord(3, 2, 0),
            Some(15)
        );
        assert_eq!(image.logical_lower_bound_complex_x, -2);
        assert_eq!(image.logical_upper_bound_real_x, 1);
    }

    #[test]
    fn logical_dimension_and_center_updates_preserve_source_sequencing() {
        let mut image = Image::default();
        image.allocate(3, 3, 1, true).unwrap();
        image.set_logical_dimensions(6, 5, 4);
        assert_eq!(image.physical_address_of_box_center_x, 1);
        image.update_looping_and_addressing();
        assert_eq!(
            (
                image.physical_address_of_box_center_x,
                image.physical_address_of_box_center_y,
                image.physical_address_of_box_center_z,
            ),
            (3, 2, 2)
        );
        assert_eq!(image.logical_lower_bound_complex_y, -2);
        image.setup_initial_values();
        assert!(image.real_values.is_empty());
        assert_eq!(image.logical_x_dimension, 0);
    }

    #[test]
    fn allocation_overloads_retain_same_dimension_storage_and_owned_cleanup() {
        let mut image = Image::default();
        image.allocate_2d(3, 2, true).unwrap();
        image.set_real_pixel_from_physical_coord(1, 1, 0, 9.);
        image.allocate(3, 2, 1, false).unwrap();
        assert!(!image.is_in_real_space);
        assert_eq!(
            image.return_real_pixel_from_physical_coord(1, 1, 0),
            Some(9.)
        );
        assert_eq!(image.return_volume_in_real_space(), 6);
        image.deallocate();
        assert!(image.real_values.is_empty());
        assert!(image.complex_values.is_empty());
        image.allocate_2d(2, 2, true).unwrap();
        assert_eq!(image.return_volume_in_real_space(), 4);
    }

    #[test]
    fn mutable_2d_slice_view_replaces_source_pointer_aliasing() {
        let mut volume = Image::default();
        volume.allocate(3, 2, 2, true).unwrap();
        let mut plane = volume.slice_2d_mut(1).unwrap();
        assert!(plane.set(2, 1, 8.));
        assert_eq!(plane.get(2, 1), Some(8.));
        assert!(!plane.set(3, 1, 0.));
        drop(plane);
        assert_eq!(
            volume.return_real_pixel_from_physical_coord(2, 1, 1),
            Some(8.)
        );
        assert!(volume.slice_2d_mut(2).is_none());
    }

    #[test]
    fn statistics_ignore_fft_padding() {
        let mut image = Image::default();
        image.allocate(4, 3, 1, true).unwrap();
        for y in 0..3 {
            for x in 0..4 {
                image.set_real_pixel_from_physical_coord(x, y, 0, (x + 4 * y) as f32);
            }
        }
        image.real_values[4] = 1_000.;
        assert_eq!(image.get_min_max(), Some((0., 11.)));
        assert_eq!(image.return_average_of_real_values(0., false), 5.5);
        assert_eq!(image.return_median_of_real_values(), Some(6.));
        assert_eq!(image.return_average_of_real_values_on_edges(), 5.5);
    }

    #[test]
    fn bilinear_interpolation_uses_logical_neighbors_not_padding() {
        let mut image = Image::default();
        image.allocate(3, 3, 1, true).unwrap();
        for y in 0..3 {
            for x in 0..3 {
                image.set_real_pixel_from_physical_coord(x, y, 0, (x + 3 * y) as f32);
            }
        }
        assert_eq!(
            image.get_real_value_by_linear_interpolation_no_bounds_check_image(0.5, 0.5),
            2.
        );
    }

    #[test]
    fn clip_into_larger_real_space_2d_centers_and_preserves_destination_padding() {
        let mut source = Image::default();
        source.allocate(2, 2, 1, true).unwrap();
        source.set_real_pixel_from_physical_coord(0, 0, 0, 1.);
        source.set_real_pixel_from_physical_coord(1, 0, 0, 2.);
        source.set_real_pixel_from_physical_coord(0, 1, 0, 3.);
        source.set_real_pixel_from_physical_coord(1, 1, 0, 4.);
        let mut destination = Image::default();
        destination.allocate(5, 5, 1, true).unwrap();
        source.clip_into_larger_real_space_2d(&mut destination, -1.);
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(0, 0, 0),
            Some(-1.)
        );
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(1, 1, 0),
            Some(1.)
        );
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(2, 1, 0),
            Some(2.)
        );
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(1, 2, 0),
            Some(3.)
        );
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(2, 2, 0),
            Some(4.)
        );
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(4, 4, 0),
            Some(-1.)
        );
        assert_eq!(destination.real_values[5], 0.);
    }

    #[test]
    fn clip_resize_consume_and_bounded_interpolation_use_owned_pixels() {
        let mut source = Image::default();
        source.allocate(3, 2, 1, true).unwrap();
        for y in 0..2 {
            for x in 0..3 {
                source.set_real_pixel_from_physical_coord(x, y, 0, (10 * y + x) as f32);
            }
        }
        let mut destination = Image::default();
        destination.allocate(3, 3, 1, true).unwrap();
        source.clip_into_real_space(&mut destination, -1., 1, 0, 0);
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(1, 1, 0),
            Some(12.)
        );
        assert_eq!(
            destination.return_real_pixel_from_physical_coord(0, 0, 0),
            Some(1.)
        );
        source.resize_real_space(5, 4, 1, -2.).unwrap();
        assert_eq!(
            source.return_real_pixel_from_physical_coord(0, 0, 0),
            Some(-2.)
        );
        assert_eq!(source.return_linear_interpolated_2d(2.5, 1.0), 1.5);
        assert_eq!(source.return_linear_interpolated_2d(-0.1, 1.0), 0.);
        let mut moved = Image::default();
        moved.consume(&mut source);
        assert_eq!(source.real_memory_allocated, 0);
        assert_eq!(moved.logical_x_dimension, 5);
    }

    #[test]
    fn identity_magnification_distortion_preserves_image() {
        let mut image = Image::default();
        image.allocate(4, 4, 1, true).unwrap();
        for y in 0..4 {
            for x in 0..4 {
                image.set_real_pixel_from_physical_coord(x, y, 0, (y * 4 + x) as f32);
            }
        }
        let before = image.real_values.clone();
        image.correct_magnification_distortion(0., 1., 1.);
        assert_eq!(image.real_values, before);
    }

    #[test]
    fn fourier_addresses_and_clip_keep_logical_frequencies() {
        let mut source = Image::default();
        source.allocate(4, 4, 1, false).unwrap();
        let source_address = source
            .return_fourier_1d_address_from_logical_coord(1, -1, 0)
            .unwrap();
        source.complex_values[source_address] = Complex32::new(3., -4.);
        let nyquist_address = source
            .return_fourier_1d_address_from_physical_coord(1, 2, 0)
            .unwrap();
        source.complex_values[nyquist_address] = Complex32::new(7., 0.);
        let mut destination = Image::default();
        destination.allocate(6, 6, 1, false).unwrap();
        source.clip_into(&mut destination, -2., false, 0., 0, 0, 0);
        assert_eq!(
            destination.return_complex_pixel_from_logical_coord(1, -1, 0, Complex32::new(0., 0.)),
            Complex32::new(3., -4.)
        );
        assert_eq!(
            destination.return_complex_pixel_from_logical_coord(1, 2, 0, Complex32::new(0., 0.)),
            Complex32::new(7., 0.)
        );
    }

    #[test]
    fn spectrum_statistics_cross_and_hermitian_operations_follow_source() {
        let mut image = Image::default();
        image.allocate(5, 5, 1, true).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                image.set_real_pixel_from_physical_coord(x, y, 0, (x + y) as f32);
            }
        }
        image.set_maximum_value_on_central_cross(1.);
        assert_eq!(
            image.return_real_pixel_from_physical_coord(2, 4, 0),
            Some(1.)
        );
        image.zero_central_pixel();
        assert_eq!(
            image.return_real_pixel_from_physical_coord(2, 2, 0),
            Some(0.)
        );
        assert!(
            image
                .compute_average_and_sigma_of_values_in_spectrum(1., 4., 0)
                .is_some()
        );
        assert!(image.fourier_component_has_explicit_hermitian_mate(0, 1, 0));
        assert!(image.fourier_component_is_explicit_hermitian_mate(0, 3, 0));
    }

    #[test]
    fn rotational_power_amplitude_and_box_spectrum_operations_are_native() {
        let mut real = Image::default();
        real.allocate(5, 5, 1, true).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                real.set_real_pixel_from_physical_coord(x, y, 0, (x + y * 5) as f32);
            }
        }
        let mut average = Curve::new();
        let mut counts = Curve::new();
        average.setup_x_axis(0., 4., 9);
        counts.setup_x_axis(0., 4., 9);
        real.compute_1d_rotational_average(&mut average, &mut counts, false);
        assert_eq!(counts.data_y.iter().sum::<f32>(), 25.);
        let mut smoothed = Image::default();
        smoothed.allocate(5, 5, 1, true).unwrap();
        real.spectrum_box_convolution(&mut smoothed, 3, 0.);
        assert_eq!(
            smoothed.return_real_pixel_from_physical_coord(2, 2, 0),
            Some(12.)
        );

        let mut fourier = Image::default();
        fourier.allocate(4, 4, 1, false).unwrap();
        let dc = fourier
            .return_fourier_1d_address_from_physical_coord(0, 0, 0)
            .unwrap();
        fourier.complex_values[dc] = Complex32::new(3., 4.);
        let mut power = Curve::new();
        let mut power_counts = Curve::new();
        power.setup_x_axis(0., 1., 5);
        power_counts.setup_x_axis(0., 1., 5);
        fourier.compute_1d_power_spectrum_curve(&mut power, &mut power_counts);
        assert_eq!(power.data_y[0], 25.);
        let mut amplitude = Image::default();
        amplitude.allocate(4, 4, 1, true).unwrap();
        fourier.compute_amplitude_spectrum_full_2d(&mut amplitude);
        assert_eq!(
            amplitude.return_real_pixel_from_physical_coord(2, 2, 0),
            Some(5.)
        );
    }

    #[test]
    fn ctf_correlation_and_precomputed_path_are_finite() {
        let ctf = Ctf::with_fitting_parameters(
            300., 2.7, 0.07, 15_000., 14_000., 20., 0.02, 0.45, -10., 1., 0.,
        );
        let mut image = Image::default();
        image.allocate(48, 48, 1, true).unwrap();
        for y in 0..48 {
            for x in 0..48 {
                image.set_real_pixel_from_physical_coord(x, y, 0, (x * 3 + y * 7) as f32);
            }
        }
        assert!(image.get_correlation_with_ctf(&ctf).is_finite());
        let (addresses, frequencies, azimuths, norm, mean) =
            image.setup_quick_correlation_with_ctf(&ctf);
        assert!(!addresses.is_empty());
        assert!(
            image
                .quick_correlation_with_ctf(&ctf, &addresses, &frequencies, &azimuths, norm, mean)
                .is_finite()
        );
    }

    #[test]
    fn normalization_and_extrema_exclude_requested_center_and_edges() {
        let mut image = Image::default();
        image.allocate(5, 5, 1, true).unwrap();
        for y in 0..5 {
            for x in 0..5 {
                image.set_real_pixel_from_physical_coord(x, y, 0, (x + y * 5) as f32);
            }
        }
        image.set_real_pixel_from_physical_coord(2, 2, 0, 100.);
        assert_eq!(image.return_maximum_value(1., 0.), 24.);
        assert_eq!(image.return_minimum_value(1., 0.), 0.);
        let mut normalized = Image::default();
        normalized.allocate(2, 2, 1, true).unwrap();
        normalized.set_to_constant(2.);
        normalized.normalize_ft();
        assert_eq!(
            normalized.return_real_pixel_from_physical_coord(0, 0, 0),
            Some(1.)
        );
        normalized.normalize_ft_and_invert_real_values();
        assert_eq!(
            normalized.return_real_pixel_from_physical_coord(0, 0, 0),
            Some(-0.5)
        );
    }

    #[test]
    fn cosine_mask_covers_uncentered_real_and_fourier_source_paths() {
        let mut uncentered = Image::default();
        uncentered.allocate(4, 4, 1, true).unwrap();
        uncentered.object_is_centred_in_box = false;
        uncentered.set_to_constant(2.);
        uncentered.cosine_mask(0.5, 1., false, Some(-1.));
        assert_eq!(
            uncentered.return_real_pixel_from_physical_coord(0, 0, 0),
            Some(-1.)
        );
        assert_eq!(
            uncentered.return_real_pixel_from_physical_coord(2, 2, 0),
            Some(-1.)
        );
        let mut fourier = Image::default();
        fourier.allocate(4, 4, 1, false).unwrap();
        fourier.complex_values.fill(Complex32::new(2., 0.));
        fourier.cosine_mask(0.1, 0.1, false, None);
        assert_eq!(fourier.complex_values[0], Complex32::new(2., 0.));
        let outer = fourier
            .return_fourier_1d_address_from_physical_coord(2, 0, 0)
            .unwrap();
        assert_eq!(fourier.complex_values[outer], Complex32::new(0., 0.));
    }

    #[test]
    fn native_fft_round_trip_preserves_padded_real_image() {
        let mut image = Image::default();
        image.allocate(4, 3, 2, true).unwrap();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    image.set_real_pixel_from_physical_coord(x, y, z, (x + y * 4 + z * 12) as f32);
                }
            }
        }
        image.forward_fft(false);
        assert!(!image.is_in_real_space);
        image.backward_fft();
        for z in 0..2 {
            for y in 0..3 {
                for x in 0..4 {
                    assert!(
                        (image
                            .return_real_pixel_from_physical_coord(x, y, z)
                            .unwrap()
                            - (x + y * 4 + z * 12) as f32)
                            .abs()
                            < 0.0001
                    );
                }
            }
        }
    }
}
