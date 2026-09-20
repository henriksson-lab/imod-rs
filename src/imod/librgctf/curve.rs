//! Translation of `IMOD/librgctf/curve.{h,cpp}`.

use super::ctf::Ctf;
use super::functions::wx_printf_fmt;
use super::types::CurvePoint;
use crate::imod::libcfshr::b3dutil::CArg;

/// C++ `Curve` (`curve.h:1`).
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub have_polynomial: bool,
    pub have_savitzky_golay: bool,
    pub number_of_points: i32,
    pub allocated_space_for_points: i32,

    pub data_x: Vec<f32>,
    pub data_y: Vec<f32>,

    pub polynomial_fit: Vec<f32>,
    pub savitzky_golay_fit: Vec<f32>,

    pub savitzky_golay_window_size: i32,
    pub savitzky_golay_polynomial_order: i32,
    pub savitzky_golay_coefficients: Vec<Vec<f32>>,

    pub polynomial_order: i32,
    pub polynomial_coefficients: Vec<f32>,
}

impl Default for Curve {
    fn default() -> Self {
        Self::new()
    }
}

impl Curve {
    /// C++ `Curve::Curve()` (`curve.cpp:8`).
    pub fn new() -> Self {
        Self {
            have_polynomial: false,
            have_savitzky_golay: false,
            number_of_points: 0,
            allocated_space_for_points: 100,
            data_x: vec![0.0; 100],
            data_y: vec![0.0; 100],
            polynomial_fit: Vec::new(),
            savitzky_golay_fit: Vec::new(),
            savitzky_golay_coefficients: Vec::new(),
            polynomial_order: 0,
            polynomial_coefficients: Vec::new(),
            savitzky_golay_polynomial_order: 0,
            savitzky_golay_window_size: 0,
        }
    }

    /// C++ `Curve::DeleteSavitzkyGolayCoefficients` (`curve.cpp:78`).
    pub fn delete_savitzky_golay_coefficients(&mut self) {
        self.savitzky_golay_coefficients.clear();
    }

    /// C++ `Curve::AllocateSavitzkyGolayCoefficients` (`curve.cpp:92`).
    pub fn allocate_savitzky_golay_coefficients(&mut self) {
        self.savitzky_golay_coefficients =
            vec![
                vec![0.0; (self.savitzky_golay_polynomial_order + 1) as usize];
                self.number_of_points as usize
            ];
    }

    /// C++ `Curve::operator = (const Curve *)` (`curve.cpp:108`).
    ///
    /// The source reallocates `polynomial_fit` and `savitzky_golay_fit` with
    /// the *old* `number_of_points` and then copies the *new* count into them,
    /// which writes past those allocations whenever the curve grows and a fit
    /// is present.  That is a latent upstream overflow; here the fit vectors
    /// are sized to the count that is copied, which is the only difference.
    pub fn assign(&mut self, other_curve: &Curve) {
        if std::ptr::eq(self, other_curve) {
            return;
        }

        let counter: i32;
        let _ = counter;

        if self.number_of_points != other_curve.number_of_points {
            self.allocated_space_for_points = other_curve.allocated_space_for_points;

            self.data_x = vec![0.0; self.allocated_space_for_points as usize];
            self.data_y = vec![0.0; self.allocated_space_for_points as usize];

            self.polynomial_order = other_curve.polynomial_order;
            self.savitzky_golay_polynomial_order = other_curve.savitzky_golay_polynomial_order;

            if self.have_polynomial {
                self.polynomial_fit.clear();
                self.polynomial_coefficients.clear();
            }

            if self.have_savitzky_golay {
                self.savitzky_golay_fit.clear();
                self.delete_savitzky_golay_coefficients();
            }

            if other_curve.have_polynomial {
                self.polynomial_fit = vec![0.0; other_curve.number_of_points as usize];
                self.polynomial_coefficients = vec![0.0; self.polynomial_order as usize];
            }

            if other_curve.have_savitzky_golay {
                self.savitzky_golay_fit = vec![0.0; other_curve.number_of_points as usize];
                let saved_points = self.number_of_points;
                self.number_of_points = other_curve.number_of_points;
                self.allocate_savitzky_golay_coefficients();
                self.number_of_points = saved_points;
            }
        } else {
            self.polynomial_order = other_curve.polynomial_order;
            self.savitzky_golay_polynomial_order = other_curve.savitzky_golay_polynomial_order;

            if self.have_polynomial != other_curve.have_polynomial {
                if self.have_polynomial {
                    self.polynomial_coefficients.clear();
                    self.polynomial_fit.clear();
                } else {
                    self.polynomial_fit = vec![0.0; self.number_of_points as usize];
                    self.polynomial_coefficients = vec![0.0; self.polynomial_order as usize];
                }
            }

            if self.have_savitzky_golay != other_curve.have_savitzky_golay {
                if self.have_savitzky_golay {
                    self.savitzky_golay_fit.clear();
                    self.delete_savitzky_golay_coefficients();
                } else {
                    self.savitzky_golay_fit = vec![0.0; self.number_of_points as usize];
                    self.allocate_savitzky_golay_coefficients();
                }
            }
        }

        self.number_of_points = other_curve.number_of_points;
        self.have_polynomial = other_curve.have_polynomial;
        self.have_savitzky_golay = other_curve.have_savitzky_golay;
        self.savitzky_golay_polynomial_order = other_curve.savitzky_golay_polynomial_order;
        self.savitzky_golay_window_size = other_curve.savitzky_golay_window_size;

        for counter in 0..self.number_of_points as usize {
            self.data_x[counter] = other_curve.data_x[counter];
            self.data_y[counter] = other_curve.data_y[counter];
        }

        if self.have_polynomial {
            for counter in 0..self.number_of_points as usize {
                self.polynomial_fit[counter] = other_curve.polynomial_fit[counter];
            }

            for counter in 0..self.polynomial_order as usize {
                self.polynomial_coefficients[counter] =
                    other_curve.polynomial_coefficients[counter];
            }
        }

        if self.have_savitzky_golay {
            for counter in 0..self.number_of_points as usize {
                self.savitzky_golay_fit[counter] = other_curve.savitzky_golay_fit[counter];
                for degree in 0..=self.savitzky_golay_polynomial_order as usize {
                    self.savitzky_golay_coefficients[counter][degree] =
                        other_curve.savitzky_golay_coefficients[counter][degree];
                }
            }
        }
    }

    /// C++ `Curve::SetupXAxis` (`curve.cpp:228`).
    pub fn setup_x_axis(
        &mut self,
        lower_bound: f32,
        upper_bound: f32,
        wanted_number_of_points: i32,
    ) {
        self.clear_data();

        for counter in 0..wanted_number_of_points {
            self.add_point(
                counter as f32 * (upper_bound - lower_bound)
                    / ((wanted_number_of_points - 1) as f32)
                    + lower_bound,
                0.0,
            );
        }
    }

    /// C++ `Curve::ZeroYData` (`curve.cpp:237`).
    pub fn zero_y_data(&mut self) {
        for counter in 0..self.number_of_points as usize {
            self.data_y[counter] = 0.0;
        }
    }

    /// C++ `Curve::AddWith` (`curve.cpp:245`).
    pub fn add_with(&mut self, other_curve: &Curve) {
        for counter in 0..self.number_of_points as usize {
            self.data_y[counter] += other_curve.data_y[counter];
        }
    }

    /// C++ `Curve::ReturnAverageValue` (`curve.cpp:256`).
    pub fn return_average_value(&self) -> f32 {
        let mut sum = 0.0f32;

        for counter in 0..self.number_of_points as usize {
            sum += self.data_y[counter];
        }

        sum / (self.number_of_points as f32)
    }

    /// C++ `Curve::ZeroAfterIndex` (`curve.cpp:269`).
    pub fn zero_after_index(&mut self, index: i32) {
        if index + 1 <= self.number_of_points {
            for counter in (index + 1).max(0) as usize..self.number_of_points as usize {
                self.data_y[counter] = 0.0;
            }
        }
    }

    /// C++ `Curve::FlattenBeforeIndex` (`curve.cpp:282`).
    pub fn flatten_before_index(&mut self, index: i32) {
        let mut index = index;
        if index > self.number_of_points {
            index = self.number_of_points;
        }
        for counter in 0..index as usize {
            self.data_y[counter] = self.data_y[index as usize];
        }
    }

    /// C++ `Curve::ResampleCurve` (`curve.cpp:291`).
    pub fn resample_curve(&mut self, input_curve: &Curve, wanted_number_of_points: i32) {
        let mut temp_curve = Curve::new();

        let mut i_x: f32;

        for i in 0..wanted_number_of_points {
            i_x = (f64::from(
                ((i * input_curve.number_of_points) as f32) / (wanted_number_of_points as f32),
            ) * (1.0 - 1.0 / f64::from((wanted_number_of_points - 1) as f32)))
                as f32;
            temp_curve.add_point(i_x, input_curve.return_linear_interpolation_from_i(i_x));
        }

        self.copy_from(&temp_curve);
    }

    /// C++ `Curve::ReturnLinearInterpolationFromI` (`curve.cpp:308`).
    pub fn return_linear_interpolation_from_i(&self, wanted_i: f32) -> f32 {
        let i = wanted_i as i32;

        let distance_below = wanted_i - i as f32;
        let distance_above = (1.0 - f64::from(distance_below)) as f32;

        if distance_below == 0.0 {
            return self.data_y[i as usize];
        }
        if distance_above == 0.0 {
            return self.data_y[(i + 1) as usize];
        }

        ((1.0 - f64::from(distance_above)) * f64::from(self.data_y[(i + 1) as usize])
            + (1.0 - f64::from(distance_below)) * f64::from(self.data_y[i as usize])) as f32
    }

    /// C++ `Curve::ReturnLinearInterpolationFromX` (`curve.cpp:344`).
    pub fn return_linear_interpolation_from_x(&self, wanted_x: f32) -> f32 {
        let mut value_to_return = 0.0f32;

        let index_of_previous_bin = self.return_index_of_nearest_previous_bin(wanted_x);

        if index_of_previous_bin == self.number_of_points - 1 {
            value_to_return = self.data_y[(self.number_of_points - 1) as usize];
        } else {
            let distance = (self.data_y.len(), ());
            let _ = distance;
            let distance = (wanted_x - self.data_x[index_of_previous_bin as usize])
                / (self.data_x[(index_of_previous_bin + 1) as usize]
                    - self.data_x[index_of_previous_bin as usize]);
            value_to_return = (f64::from(value_to_return)
                + f64::from(self.data_y[index_of_previous_bin as usize])
                    * (1.0 - f64::from(distance))) as f32;
            value_to_return = (f64::from(value_to_return)
                + f64::from(self.data_y[(index_of_previous_bin + 1) as usize])
                    * f64::from(distance)) as f32;
        }
        value_to_return
    }

    /// C++ `Curve::ComputeMaximumValueAndMode` (`curve.cpp:365`).
    pub fn compute_maximum_value_and_mode(&self, maximum_value: &mut f32, mode: &mut f32) {
        *maximum_value = -f32::MAX;

        for counter in 0..self.number_of_points as usize {
            if self.data_y[counter] > *maximum_value {
                *maximum_value = self.data_y[counter];
                *mode = self.data_x[counter];
            }
        }
    }

    /// C++ `Curve::ReturnFullWidthAtGivenValue` (`curve.cpp:381`).
    pub fn return_full_width_at_given_value(&self, wanted_value: f32) -> f32 {
        let mut first_bin_above_value: i32 = -1;
        let mut last_bin_above_value: i32 = -1;

        for counter in 0..self.number_of_points {
            if first_bin_above_value == -1 && self.data_y[counter as usize] > wanted_value {
                first_bin_above_value = counter;
            }
            if last_bin_above_value == -1
                && first_bin_above_value != -1
                && self.data_y[counter as usize] < wanted_value
            {
                last_bin_above_value = counter - 1;
            }
        }

        self.data_x[(last_bin_above_value + 1) as usize]
            - self.data_x[first_bin_above_value as usize]
    }

    /// C++ `Curve::ReturnMaximumValue` (`curve.cpp:401`).
    pub fn return_maximum_value(&self) -> f32 {
        let mut maximum_value = 0.0f32;
        let mut mode = 0.0f32;
        self.compute_maximum_value_and_mode(&mut maximum_value, &mut mode);
        maximum_value
    }

    /// C++ `Curve::ReturnMode` (`curve.cpp:408`).
    pub fn return_mode(&self) -> f32 {
        let mut maximum_value = 0.0f32;
        let mut mode = 0.0f32;
        self.compute_maximum_value_and_mode(&mut maximum_value, &mut mode);
        mode
    }

    /// C++ `Curve::NormalizeMaximumValue` (`curve.cpp:416`).
    pub fn normalize_maximum_value(&mut self) {
        let maximum_value = self.return_maximum_value();

        if maximum_value > 0.0 {
            let factor = (1.0 / f64::from(maximum_value)) as f32;
            for counter in 0..self.number_of_points as usize {
                self.data_y[counter] *= factor;
            }
        }
    }

    /// C++ `Curve::SquareRoot` (`curve.cpp:436`).
    pub fn square_root(&mut self) {
        for counter in 0..self.number_of_points as usize {
            self.data_y[counter] = self.data_y[counter].sqrt();
        }
    }

    /// C++ `Curve::ReturnValueAtXUsingLinearInterpolation` (`curve.cpp:452`).
    pub fn return_value_at_x_using_linear_interpolation(
        &self,
        wanted_x: f32,
        value_to_add: f32,
        assume_linear_x: bool,
    ) -> CurvePoint {
        let index_of_previous_bin: i32;
        let mut return_value = CurvePoint::default();
        if assume_linear_x {
            index_of_previous_bin =
                ((wanted_x - self.data_x[0]) / (self.data_x[1] - self.data_x[0])) as i32;
        } else {
            index_of_previous_bin = self.return_index_of_nearest_previous_bin(wanted_x);
        }

        if index_of_previous_bin == self.number_of_points - 1 {
            return_value.index_m = index_of_previous_bin;
            return_value.index_n = index_of_previous_bin;
            return_value.value_m = value_to_add;
            return_value.value_n = 0.0;
        } else {
            let distance = (wanted_x - self.data_x[index_of_previous_bin as usize])
                / (self.data_x[(index_of_previous_bin + 1) as usize]
                    - self.data_x[index_of_previous_bin as usize]);
            return_value.index_m = index_of_previous_bin;
            return_value.index_n = index_of_previous_bin + 1;
            return_value.value_m = (f64::from(value_to_add) * (1.0 - f64::from(distance))) as f32;
            return_value.value_n = value_to_add * distance;
        }

        return_value
    }

    /// C++ `Curve::AddValueAtXUsingLinearInterpolation` (`curve.cpp:484`).
    pub fn add_value_at_x_using_linear_interpolation(
        &mut self,
        wanted_x: f32,
        value_to_add: f32,
        assume_linear_x: bool,
    ) {
        let return_value = self.return_value_at_x_using_linear_interpolation(
            wanted_x,
            value_to_add,
            assume_linear_x,
        );
        self.data_y[return_value.index_m as usize] += return_value.value_m;
        self.data_y[return_value.index_n as usize] += return_value.value_n;
    }

    /// C++ `Curve::AddValueAtXUsingNearestNeighborInterpolation` (`curve.cpp:495`).
    pub fn add_value_at_x_using_nearest_neighbor_interpolation(
        &mut self,
        wanted_x: f32,
        value_to_add: f32,
    ) {
        let index = self.return_index_of_nearest_point_from_x(wanted_x);
        self.data_y[index as usize] += value_to_add;
    }

    /// C++ `Curve::PrintToStandardOut` (`curve.cpp:505`).
    pub fn print_to_standard_out(&self) {
        for i in 0..self.number_of_points as usize {
            wx_printf_fmt(
                "%f %f\n",
                &[
                    CArg::Dbl(f64::from(self.data_x[i])),
                    CArg::Dbl(f64::from(self.data_y[i])),
                ],
            );
        }
    }

    /// C++ `Curve::CopyFrom` (`curve.cpp:533`).
    pub fn copy_from(&mut self, other_curve: &Curve) {
        self.assign(other_curve);
    }

    /// C++ `Curve::GetXMinMax` (`curve.cpp:539`).
    pub fn get_x_min_max(&self, min_value: &mut f32, max_value: &mut f32) {
        *min_value = f32::MAX;
        *max_value = -f32::MAX;

        for point_counter in 0..self.number_of_points as usize {
            *min_value = if self.data_x[point_counter] < *min_value {
                self.data_x[point_counter]
            } else {
                *min_value
            };
            *max_value = if *max_value < self.data_x[point_counter] {
                self.data_x[point_counter]
            } else {
                *max_value
            };
        }
    }

    /// C++ `Curve::GetYMinMax` (`curve.cpp:551`).
    pub fn get_y_min_max(&self, min_value: &mut f32, max_value: &mut f32) {
        *min_value = f32::MAX;
        *max_value = -f32::MAX;

        for point_counter in 0..self.number_of_points as usize {
            *min_value = if self.data_y[point_counter] < *min_value {
                self.data_y[point_counter]
            } else {
                *min_value
            };
            *max_value = if *max_value < self.data_y[point_counter] {
                self.data_y[point_counter]
            } else {
                *max_value
            };
        }
    }

    /// C++ `Curve::CheckMemory` (`curve.cpp:562`).
    pub fn check_memory(&mut self) {
        if self.number_of_points >= self.allocated_space_for_points {
            // reallocate..

            if self.allocated_space_for_points < 10000 {
                self.allocated_space_for_points *= 2;
            } else {
                self.allocated_space_for_points += 10000;
            }

            self.data_x
                .resize(self.allocated_space_for_points as usize, 0.0);
            self.data_y
                .resize(self.allocated_space_for_points as usize, 0.0);
        }
    }

    /// C++ `Curve::AddPoint` (`curve.cpp:587`).
    pub fn add_point(&mut self, x_value: f32, y_value: f32) {
        // check memory

        self.check_memory();

        // add the point

        self.data_x[self.number_of_points as usize] = x_value;
        self.data_y[self.number_of_points as usize] = y_value;

        self.number_of_points += 1;
    }

    /// C++ `Curve::ClearData` (`curve.cpp:600`).
    pub fn clear_data(&mut self) {
        self.number_of_points = 0;

        if self.have_polynomial {
            self.polynomial_fit.clear();
            self.polynomial_coefficients.clear();

            self.have_polynomial = false;
        }

        if self.have_savitzky_golay {
            self.savitzky_golay_fit.clear();

            self.have_savitzky_golay = false;
        }
    }

    /// C++ `Curve::MultiplyByConstant` (`curve.cpp:619`).
    pub fn multiply_by_constant(&mut self, constant_to_multiply_by: f32) {
        for counter in 0..self.number_of_points as usize {
            self.data_y[counter] *= constant_to_multiply_by;
        }
    }

    /// C++ `Curve::ApplyCTF` (`curve.cpp:630`).
    pub fn apply_ctf(&mut self, ctf_to_apply: Ctf, azimuth_in_radians: f32) {
        for counter in 0..self.number_of_points as usize {
            self.data_y[counter] *=
                ctf_to_apply.evaluate(self.data_x[counter].powf(2.0), azimuth_in_radians);
        }
    }

    /// C++ `Curve::ReturnSavitzkyGolayInterpolationFromX` (`curve.cpp:640`).
    pub fn return_savitzky_golay_interpolation_from_x(&self, wanted_x: f32) -> f32 {
        // Find the nearest data point to the wanted_x
        let index_of_nearest_point = self.return_index_of_nearest_point_from_x(wanted_x);

        // Evaluate the polynomial defined at the nearest point.
        let mut y = f64::from(self.savitzky_golay_coefficients[index_of_nearest_point as usize][0]);
        for order in 1..=self.savitzky_golay_polynomial_order as usize {
            y += f64::from(wanted_x).powi(order as i32)
                * f64::from(
                    self.savitzky_golay_coefficients[index_of_nearest_point as usize][order],
                );
        }

        y as f32
    }

    /// C++ `Curve::ReturnIndexOfNearestPointFromX` (`curve.cpp:657`).
    pub fn return_index_of_nearest_point_from_x(&self, wanted_x: f32) -> i32 {
        let mut index_of_nearest_point = 0i32;
        let mut counter = 0i32;
        let mut distance_to_current_point = wanted_x - self.data_x[counter as usize];
        let mut distance_to_nearest_point = distance_to_current_point;
        counter = 1;
        while counter < self.number_of_points {
            distance_to_current_point = wanted_x - self.data_x[counter as usize];
            if distance_to_current_point.abs() <= distance_to_nearest_point.abs() {
                distance_to_nearest_point = distance_to_current_point;
                index_of_nearest_point = counter;
            } else {
                break;
            }
            counter += 1;
        }
        index_of_nearest_point
    }

    /// C++ `Curve::ReturnIndexOfNearestPreviousBin` (`curve.cpp:681`).
    pub fn return_index_of_nearest_previous_bin(&self, wanted_x: f32) -> i32 {
        if wanted_x < self.data_x[0] {
            0
        } else if wanted_x >= self.data_x[(self.number_of_points - 1) as usize] {
            self.number_of_points - 1
        } else {
            for counter in 0..self.number_of_points - 1 {
                if wanted_x >= self.data_x[counter as usize]
                    && wanted_x < self.data_x[(counter + 1) as usize]
                {
                    return counter;
                }
            }
            // Should never get here
            0
        }
    }

    /// C++ `Curve::FitSavitzkyGolayToData` (`curve.cpp:707`).
    pub fn fit_savitzky_golay_to_data(
        &mut self,
        wanted_window_size: i32,
        wanted_polynomial_order: i32,
    ) {
        let mut pixel_counter: i32;
        let mut polynomial_counter: i32;

        let end_start: i32;

        let half_pixel = wanted_window_size / 2;

        let mut fit_array_x = vec![0.0f32; wanted_window_size as usize];
        let mut fit_array_y = vec![0.0f32; wanted_window_size as usize];
        let mut output_fit_array = vec![0.0f32; wanted_window_size as usize];

        // Remember the polymomal order and the window size
        self.savitzky_golay_polynomial_order = wanted_polynomial_order;
        self.savitzky_golay_window_size = wanted_window_size;

        // Allocate array of coefficient arrays, to be kept in memory for later use
        if !self.savitzky_golay_coefficients.is_empty() {
            self.delete_savitzky_golay_coefficients();
        }
        // Allocate memory for smooth y values
        if self.have_savitzky_golay {
            self.savitzky_golay_fit.clear();
        }

        self.allocate_savitzky_golay_coefficients();
        self.savitzky_golay_fit = vec![0.0; self.number_of_points as usize];
        self.have_savitzky_golay = true;

        // loop over all the points..

        pixel_counter = 0;
        while pixel_counter < self.number_of_points - 2 * half_pixel {
            // for this pixel, extract the window, fit the polynomial, and copy
            // the average into the output array
            for polynomial_counter in 0..wanted_window_size {
                fit_array_x[polynomial_counter as usize] =
                    self.data_x[(pixel_counter + polynomial_counter) as usize];
                fit_array_y[polynomial_counter as usize] =
                    self.data_y[(pixel_counter + polynomial_counter) as usize];
            }

            // fit a polynomial to this data..

            ls_poly(
                &fit_array_x,
                &fit_array_y,
                wanted_window_size,
                wanted_polynomial_order,
                &mut output_fit_array,
                &mut self.savitzky_golay_coefficients[(half_pixel + pixel_counter) as usize],
            );

            // take the middle pixel, and put it into the output array..

            self.savitzky_golay_fit[(half_pixel + pixel_counter) as usize] =
                output_fit_array[half_pixel as usize];

            pixel_counter += 1;
        }

        // now we need to take care of the ends - first the start..
        // DNM: Need to take actual points beyond the end of the fitted points in the middle
        for polynomial_counter in 0..wanted_window_size {
            fit_array_x[polynomial_counter as usize] = self.data_x[polynomial_counter as usize];

            if polynomial_counter < half_pixel
                || polynomial_counter >= self.number_of_points - half_pixel
            {
                fit_array_y[polynomial_counter as usize] = self.data_y[polynomial_counter as usize];
            } else {
                fit_array_y[polynomial_counter as usize] =
                    self.savitzky_golay_fit[polynomial_counter as usize];
            }
        }

        // fit a polynomial to this data..

        ls_poly(
            &fit_array_x,
            &fit_array_y,
            wanted_window_size,
            wanted_polynomial_order,
            &mut output_fit_array,
            &mut self.savitzky_golay_coefficients[(half_pixel - 1) as usize],
        );

        // copy the required data back..
        for pixel_counter in 0..half_pixel - 1 {
            for polynomial_counter in 0..=self.savitzky_golay_polynomial_order {
                self.savitzky_golay_coefficients[pixel_counter as usize]
                    [polynomial_counter as usize] = self.savitzky_golay_coefficients
                    [(half_pixel - 1) as usize][polynomial_counter as usize];
            }
        }
        for polynomial_counter in 0..half_pixel {
            self.savitzky_golay_fit[polynomial_counter as usize] =
                output_fit_array[polynomial_counter as usize];
        }

        // now the end..

        end_start = self.number_of_points - wanted_window_size;
        pixel_counter = 0;
        polynomial_counter = end_start;
        while polynomial_counter < self.number_of_points {
            fit_array_x[pixel_counter as usize] = self.data_x[polynomial_counter as usize];

            if pixel_counter > half_pixel {
                fit_array_y[pixel_counter as usize] = self.data_y[polynomial_counter as usize];
            } else {
                fit_array_y[pixel_counter as usize] =
                    self.savitzky_golay_fit[polynomial_counter as usize];
            }

            pixel_counter += 1;
            polynomial_counter += 1;
        }

        // fit a polynomial to this data..

        ls_poly(
            &fit_array_x,
            &fit_array_y,
            wanted_window_size,
            wanted_polynomial_order,
            &mut output_fit_array,
            &mut self.savitzky_golay_coefficients[(self.number_of_points - half_pixel) as usize],
        );

        // copy the required data back..

        for pixel_counter in self.number_of_points - half_pixel + 1..self.number_of_points - 1 {
            for polynomial_counter in 0..=self.savitzky_golay_polynomial_order {
                self.savitzky_golay_coefficients[pixel_counter as usize]
                    [polynomial_counter as usize] = self.savitzky_golay_coefficients
                    [(self.number_of_points - half_pixel) as usize]
                    [polynomial_counter as usize];
            }
        }

        pixel_counter = half_pixel + 1;
        polynomial_counter = self.number_of_points - half_pixel;
        while polynomial_counter < self.number_of_points {
            self.savitzky_golay_fit[polynomial_counter as usize] =
                output_fit_array[pixel_counter as usize];
            pixel_counter += 1;
            polynomial_counter += 1;
        }
    }

    /// C++ `Curve::FitPolynomialToData` (`curve.cpp:857`).
    pub fn fit_polynomial_to_data(&mut self, wanted_polynomial_order: i32) {
        if self.have_polynomial {
            self.polynomial_coefficients.clear();
            self.polynomial_fit.clear();
        }

        self.polynomial_fit = vec![0.0; self.number_of_points as usize];
        self.polynomial_order = wanted_polynomial_order;
        self.polynomial_coefficients = vec![0.0; (self.polynomial_order + 1) as usize];
        self.have_polynomial = true;

        // weird old code to do the fit
        let data_x = self.data_x.clone();
        let data_y = self.data_y.clone();
        ls_poly(
            &data_x,
            &data_y,
            self.number_of_points,
            self.polynomial_order,
            &mut self.polynomial_fit,
            &mut self.polynomial_coefficients,
        );
    }

    /// C++ `Curve::Reciprocal` (`curve.cpp:872`).
    pub fn reciprocal(&mut self) {
        for counter in 0..self.number_of_points as usize {
            if self.data_y[counter] != 0.0 {
                self.data_y[counter] = (1.0 / f64::from(self.data_y[counter])) as f32;
            }
        }
    }

    /// C++ `Curve::MultiplyXByConstant` (`curve.cpp:888`).
    pub fn multiply_x_by_constant(&mut self, constant_to_multiply_by: f32) {
        for counter in 0..self.number_of_points as usize {
            self.data_x[counter] *= constant_to_multiply_by;
        }
    }
}

/// C++ free function `LS_POLY` (`curve.cpp:1057`).
///
/// Least-squares polynomial fitting with Forsythe orthogonal polynomials, from
/// Ruckdeschel's *BASIC Scientific Subroutines* by way of J-P Moreau.  The
/// source's `goto` structure is kept: the `e10` block is the outer loop, `e15`
/// the inner one, and `e50` is unreachable because the caller's `e1` is always
/// zero.  `output_coefficients` receives `order_of_polynomial + 1` values.
pub fn ls_poly(
    x_data: &[f32],
    y_data: &[f32],
    number_of_points: i32,
    order_of_polynomial: i32,
    output_smoothed_curve: &mut [f32],
    output_coefficients: &mut [f32],
) {
    let mut a = vec![0.0f64; (order_of_polynomial + 2) as usize];
    let mut b = vec![0.0f64; (order_of_polynomial + 2) as usize];
    let mut c = vec![0.0f64; (order_of_polynomial + 3) as usize];
    let mut c2 = vec![0.0f64; (order_of_polynomial + 2) as usize];
    let mut f = vec![0.0f64; (order_of_polynomial + 2) as usize];

    let mut v = vec![0.0f64; (number_of_points + 1) as usize];
    let mut d = vec![0.0f64; (number_of_points + 1) as usize];
    let mut e = vec![0.0f64; (number_of_points + 1) as usize];
    let mut x = vec![0.0f64; (number_of_points + 1) as usize];
    let mut y = vec![0.0f64; (number_of_points + 1) as usize];

    let mut l: i32;
    let n = number_of_points;
    let mut m = order_of_polynomial;

    let e1 = 0.0f64;
    let mut dd: f64;
    let mut vv: f64;

    let mut i: i32;
    let mut l2: i32;
    let n1: i32;

    let mut a1: f64;
    let mut a2: f64;
    let mut b1: f64;
    let mut b2: f64;
    let mut c1: f64;
    let mut d1: f64;
    let mut f1: f64;
    let mut f2: f64;
    let mut v1: f64;
    let mut v2: f64;
    let w: f64;

    l = 0;
    n1 = m + 1;
    v1 = 1e7;

    for i in 0..number_of_points as usize {
        x[i + 1] = f64::from(x_data[i]);
        y[i + 1] = f64::from(y_data[i]);
    }

    // Initialize the arrays
    for i in 1..n1 + 1 {
        a[i as usize] = 0.0;
        b[i as usize] = 0.0;
        f[i as usize] = 0.0;
    }
    for i in 1..n + 1 {
        v[i as usize] = 0.0;
        d[i as usize] = 0.0;
    }
    d1 = f64::from(n).sqrt();
    w = d1;
    for i in 1..n + 1 {
        e[i as usize] = 1.0 / w;
    }
    f1 = d1;
    a1 = 0.0;
    for i in 1..n + 1 {
        a1 += x[i as usize] * e[i as usize] * e[i as usize];
    }
    c1 = 0.0;
    for i in 1..n + 1 {
        c1 += y[i as usize] * e[i as usize];
    }
    b[1] = 1.0 / f1;
    f[1] = b[1] * c1;
    for i in 1..n + 1 {
        v[i as usize] += e[i as usize] * c1;
    }
    m = 1;

    let mut aborted = false;
    loop {
        // e10: Save latest results
        for i in 1..l + 1 {
            c2[i as usize] = c[i as usize];
        }
        l2 = l;
        v2 = v1;
        f2 = f1;
        a2 = a1;
        f1 = 0.0;
        for i in 1..n + 1 {
            b1 = e[i as usize];
            e[i as usize] = (x[i as usize] - a2) * e[i as usize] - f2 * d[i as usize];
            d[i as usize] = b1;
            f1 += e[i as usize] * e[i as usize];
        }
        f1 = f1.sqrt();
        for i in 1..n + 1 {
            e[i as usize] /= f1;
        }
        a1 = 0.0;
        for i in 1..n + 1 {
            a1 += x[i as usize] * e[i as usize] * e[i as usize];
        }
        c1 = 0.0;
        for i in 1..n + 1 {
            c1 += e[i as usize] * y[i as usize];
        }
        m += 1;
        i = 0;
        loop {
            // e15
            l = m - i;
            b2 = b[l as usize];
            d1 = 0.0;
            if l > 1 {
                d1 = b[(l - 1) as usize];
            }
            d1 = d1 - a2 * b[l as usize] - f2 * a[l as usize];
            b[l as usize] = d1 / f1;
            a[l as usize] = b2;
            i += 1;
            if i == m {
                break;
            }
        }
        for i in 1..n + 1 {
            v[i as usize] += e[i as usize] * c1;
        }
        for i in 1..n1 + 1 {
            f[i as usize] += b[i as usize] * c1;
            c[i as usize] = f[i as usize];
        }
        vv = 0.0;
        for i in 1..n + 1 {
            vv += (v[i as usize] - y[i as usize]) * (v[i as usize] - y[i as usize]);
        }
        // Note the division is by the number of degrees of freedom
        vv = (vv / f64::from(n - l - 1)).sqrt();
        l = m;
        if e1 != 0.0 {
            // Test for minimal improvement
            if (v1 - vv).abs() / vv < e1 || e1 * vv > e1 * v1 {
                // e50: Aborted sequence, recover last values
                l = l2;
                vv = v2;
                for i in 1..l + 1 {
                    c[i as usize] = c2[i as usize];
                }
                aborted = true;
                let _ = vv;
                break;
            }
            v1 = vv;
        }
        // e20
        if m == n1 {
            dd = vv;
            let _ = dd;
            break;
        }
    }
    let _ = aborted;
    let _ = v1;
    let _ = l2;
    let _ = v2;

    // e30: Shift the c[i] down, so c(0) is the constant term
    for i in 1..l + 1 {
        c[(i - 1) as usize] = c[i as usize];
    }
    c[l as usize] = 0.0;
    // l is the order of the polynomial fitted
    l -= 1;
    let _ = l;

    for i in 0..number_of_points as usize {
        output_smoothed_curve[i] = v[i + 1] as f32;
    }

    // coefficient 0: constant, 1: linear, 2: square, ...
    for i in 0..=order_of_polynomial as usize {
        output_coefficients[i] = c[i] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::Curve;

    #[test]
    fn setup_x_axis_and_linear_interpolation_follow_source() {
        let mut curve = Curve::new();
        curve.setup_x_axis(0.0, 1.0, 5);
        assert_eq!(curve.number_of_points, 5);
        assert_eq!(curve.data_x[4], 1.0);
        curve.add_value_at_x_using_linear_interpolation(0.25, 1.0, true);
        assert_eq!(curve.data_y[1], 1.0);
        assert_eq!(curve.return_index_of_nearest_previous_bin(0.3), 1);
    }

    #[test]
    fn ls_poly_recovers_an_exact_quadratic() {
        let x: Vec<f32> = (0..7).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|x| 3.0 - 2.0 * x + 0.5 * x * x).collect();
        let mut fit = vec![0.0f32; 7];
        let mut coefficients = vec![0.0f32; 3];
        super::ls_poly(&x, &y, 7, 2, &mut fit, &mut coefficients);
        assert!((coefficients[0] - 3.0).abs() < 1e-3);
        assert!((coefficients[1] + 2.0).abs() < 1e-3);
        assert!((coefficients[2] - 0.5).abs() < 1e-3);
        assert!((fit[3] - y[3]).abs() < 1e-3);
    }

    #[test]
    fn savitzky_golay_fit_smooths_and_interpolates() {
        let mut curve = Curve::new();
        for i in 0..20 {
            curve.add_point(i as f32, (i as f32) * 0.5 + 1.0);
        }
        curve.fit_savitzky_golay_to_data(7, 2);
        assert_eq!(curve.savitzky_golay_fit.len(), 20);
        assert!((curve.return_savitzky_golay_interpolation_from_x(10.0) - 6.0).abs() < 1e-2);
    }
}
