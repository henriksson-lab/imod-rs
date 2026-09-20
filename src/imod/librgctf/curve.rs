//! Translation of `IMOD/librgctf/curve.{h,cpp}`.

use crate::imod::librgctf::types::CurvePoint;

/// A sampled one-dimensional curve and its optional polynomial fits.
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub have_polynomial: bool,
    pub have_savitzky_golay: bool,
    pub number_of_points: usize,
    pub allocated_space_for_points: usize,
    pub data_x: Vec<f32>,
    pub data_y: Vec<f32>,
    pub polynomial_fit: Vec<f32>,
    pub savitzky_golay_fit: Vec<f32>,
    pub savitzky_golay_window_size: usize,
    pub savitzky_golay_polynomial_order: usize,
    pub savitzky_golay_coefficients: Vec<Vec<f32>>,
    pub polynomial_order: usize,
    pub polynomial_coefficients: Vec<f32>,
}

impl Curve {
    /// C++ `Curve::Curve()`.
    pub fn new() -> Self {
        Self {
            have_polynomial: false,
            have_savitzky_golay: false,
            number_of_points: 0,
            allocated_space_for_points: 100,
            data_x: Vec::with_capacity(100),
            data_y: Vec::with_capacity(100),
            polynomial_fit: Vec::new(),
            savitzky_golay_fit: Vec::new(),
            savitzky_golay_window_size: 0,
            savitzky_golay_polynomial_order: 0,
            savitzky_golay_coefficients: Vec::new(),
            polynomial_order: 0,
            polynomial_coefficients: Vec::new(),
        }
    }

    /// C++ `Curve::SetupXAxis`.
    pub fn setup_x_axis(
        &mut self,
        lower_bound: f32,
        upper_bound: f32,
        wanted_number_of_points: usize,
    ) {
        assert!(
            wanted_number_of_points > 1,
            "wanted_number_of_points is smaller than 2"
        );
        self.clear_data();
        for counter in 0..wanted_number_of_points {
            self.add_point(
                counter as f32 * (upper_bound - lower_bound) / (wanted_number_of_points - 1) as f32
                    + lower_bound,
                0.0,
            );
        }
    }

    /// C++ `Curve::ZeroYData`.
    pub fn zero_y_data(&mut self) {
        self.data_y.fill(0.0);
    }

    /// C++ `Curve::ReturnLinearInterpolationFromI`.
    pub fn return_linear_interpolation_from_i(&self, wanted_i: f32) -> f32 {
        assert!(!self.data_y.is_empty(), "No points to interpolate");
        assert!(
            wanted_i <= (self.number_of_points - 1) as f32,
            "Index too high"
        );
        assert!(wanted_i >= 0.0, "Index too low");
        let index = wanted_i as usize;
        let distance_below = wanted_i - index as f32;
        let distance_above = 1.0 - distance_below;
        if distance_below == 0.0 {
            return self.data_y[index];
        }
        if distance_above == 0.0 {
            return self.data_y[index + 1];
        }
        (1.0 - distance_above) * self.data_y[index + 1]
            + (1.0 - distance_below) * self.data_y[index]
    }

    /// C++ `Curve::ComputeMaximumValueAndMode`.
    pub fn compute_maximum_value_and_mode(&self) -> (f32, f32) {
        assert!(!self.data_y.is_empty(), "No points in curve");
        let mut maximum_value = -f32::MAX;
        let mut mode = 0.0;
        for (&x, &y) in self.data_x.iter().zip(&self.data_y) {
            if y > maximum_value {
                maximum_value = y;
                mode = x;
            }
        }
        (maximum_value, mode)
    }

    pub fn return_maximum_value(&self) -> f32 {
        self.compute_maximum_value_and_mode().0
    }

    /// C++ `Curve::ReturnValueAtXUsingLinearInterpolation`.
    pub fn return_value_at_x_using_linear_interpolation(
        &self,
        wanted_x: f32,
        value_to_add: f32,
        assume_linear_x: bool,
    ) -> CurvePoint {
        assert!(!self.data_x.is_empty(), "No points in curve");
        let extent = self.data_x[self.number_of_points - 1] - self.data_x[0];
        assert!(
            wanted_x >= self.data_x[0] - extent * 0.01
                && wanted_x <= self.data_x[self.number_of_points - 1] + extent * 0.01,
            "Wanted X falls outside curve range"
        );
        let index_m = if assume_linear_x {
            ((wanted_x - self.data_x[0]) / (self.data_x[1] - self.data_x[0])) as usize
        } else {
            self.return_index_of_nearest_previous_bin(wanted_x)
        };
        if index_m == self.number_of_points - 1 {
            return CurvePoint {
                index_m: index_m as i32,
                index_n: index_m as i32,
                value_m: value_to_add,
                value_n: 0.0,
            };
        }
        let distance =
            (wanted_x - self.data_x[index_m]) / (self.data_x[index_m + 1] - self.data_x[index_m]);
        CurvePoint {
            index_m: index_m as i32,
            index_n: (index_m + 1) as i32,
            value_m: value_to_add * (1.0 - distance),
            value_n: value_to_add * distance,
        }
    }

    /// C++ `Curve::AddValueAtXUsingLinearInterpolation`.
    pub fn add_value_at_x_using_linear_interpolation(
        &mut self,
        wanted_x: f32,
        value_to_add: f32,
        assume_linear_x: bool,
    ) {
        let point = self.return_value_at_x_using_linear_interpolation(
            wanted_x,
            value_to_add,
            assume_linear_x,
        );
        self.data_y[point.index_m as usize] += point.value_m;
        self.data_y[point.index_n as usize] += point.value_n;
    }

    /// C++ `Curve::CopyFrom`.
    pub fn copy_from(&mut self, other_curve: &Self) {
        self.clone_from(other_curve);
    }

    /// C++ `Curve::CheckMemory`; `Vec` grows safely, while retaining the
    /// source-visible capacity count for clients that inspect it.
    pub fn check_memory(&mut self) {
        if self.number_of_points >= self.allocated_space_for_points {
            self.allocated_space_for_points = if self.allocated_space_for_points < 10_000 {
                self.allocated_space_for_points * 2
            } else {
                self.allocated_space_for_points + 10_000
            };
            self.data_x.reserve(
                self.allocated_space_for_points
                    .saturating_sub(self.data_x.capacity()),
            );
            self.data_y.reserve(
                self.allocated_space_for_points
                    .saturating_sub(self.data_y.capacity()),
            );
        }
    }

    /// C++ `Curve::AddPoint`.
    pub fn add_point(&mut self, x_value: f32, y_value: f32) {
        self.check_memory();
        self.data_x.push(x_value);
        self.data_y.push(y_value);
        self.number_of_points = self.data_x.len();
    }

    /// C++ `Curve::ClearData`.
    pub fn clear_data(&mut self) {
        self.data_x.clear();
        self.data_y.clear();
        self.number_of_points = 0;
        self.have_polynomial = false;
        self.polynomial_fit.clear();
        self.polynomial_coefficients.clear();
        self.have_savitzky_golay = false;
        self.savitzky_golay_fit.clear();
        self.savitzky_golay_coefficients.clear();
    }

    pub fn multiply_by_constant(&mut self, constant: f32) {
        for value in &mut self.data_y {
            *value *= constant;
        }
    }

    /// C++ `Curve::ReturnSavitzkyGolayInterpolationFromX`.
    pub fn return_savitzky_golay_interpolation_from_x(&self, wanted_x: f32) -> f32 {
        assert!(self.have_savitzky_golay, "No Savitzky-Golay fit");
        let index = self.return_index_of_nearest_point_from_x(wanted_x);
        self.savitzky_golay_coefficients[index]
            .iter()
            .enumerate()
            .map(|(order, coefficient)| coefficient * wanted_x.powi(order as i32))
            .sum()
    }

    /// C++ `Curve::ReturnIndexOfNearestPointFromX`.
    pub fn return_index_of_nearest_point_from_x(&self, wanted_x: f32) -> usize {
        assert!(!self.data_x.is_empty(), "No points in curve");
        let mut index = 0;
        let mut distance = wanted_x - self.data_x[0];
        for (candidate, &x) in self.data_x.iter().enumerate().skip(1) {
            let candidate_distance = wanted_x - x;
            if candidate_distance.abs() <= distance.abs() {
                distance = candidate_distance;
                index = candidate;
            } else {
                break;
            }
        }
        index
    }

    /// C++ `Curve::ReturnIndexOfNearestPreviousBin`.
    pub fn return_index_of_nearest_previous_bin(&self, wanted_x: f32) -> usize {
        assert!(!self.data_x.is_empty(), "No points in curve");
        let extent = self.data_x[self.number_of_points - 1] - self.data_x[0];
        assert!(
            wanted_x >= self.data_x[0] - extent * 0.01
                && wanted_x <= self.data_x[self.number_of_points - 1] + extent * 0.01,
            "Wanted X falls outside curve range"
        );
        if wanted_x < self.data_x[0] {
            return 0;
        }
        if wanted_x >= self.data_x[self.number_of_points - 1] {
            return self.number_of_points - 1;
        }
        self.data_x
            .windows(2)
            .position(|pair| wanted_x >= pair[0] && wanted_x < pair[1])
            .expect("Curve X axis is not ordered")
    }

    /// C++ `Curve::FitSavitzkyGolayToData`.
    pub fn fit_savitzky_golay_to_data(
        &mut self,
        wanted_window_size: usize,
        wanted_polynomial_order: usize,
    ) {
        assert!(wanted_window_size % 2 == 1, "Window must be odd");
        assert!(
            wanted_window_size < self.number_of_points,
            "Window size is larger than the number of points"
        );
        assert!(
            wanted_polynomial_order < wanted_window_size,
            "polynomial order is larger than the window size"
        );
        let half = wanted_window_size / 2;
        assert!(half > 0, "Window must be at least three");
        self.savitzky_golay_polynomial_order = wanted_polynomial_order;
        self.savitzky_golay_window_size = wanted_window_size;
        self.allocate_savitzky_golay_coefficients();
        self.savitzky_golay_fit = vec![0.0; self.number_of_points];
        self.have_savitzky_golay = true;
        for pixel in 0..self.number_of_points - 2 * half {
            let mut fitted = vec![0.0; wanted_window_size];
            ls_poly(
                &self.data_x[pixel..pixel + wanted_window_size],
                &self.data_y[pixel..pixel + wanted_window_size],
                wanted_polynomial_order,
                &mut fitted,
                &mut self.savitzky_golay_coefficients[half + pixel],
            );
            self.savitzky_golay_fit[half + pixel] = fitted[half];
        }
        let mut x = self.data_x[..wanted_window_size].to_vec();
        let mut y = Vec::with_capacity(wanted_window_size);
        for index in 0..wanted_window_size {
            y.push(if index < half || index >= self.number_of_points - half {
                self.data_y[index]
            } else {
                self.savitzky_golay_fit[index]
            });
        }
        let mut fitted = vec![0.0; wanted_window_size];
        ls_poly(
            &x,
            &y,
            wanted_polynomial_order,
            &mut fitted,
            &mut self.savitzky_golay_coefficients[half - 1],
        );
        for pixel in 0..half - 1 {
            self.savitzky_golay_coefficients[pixel] =
                self.savitzky_golay_coefficients[half - 1].clone();
        }
        self.savitzky_golay_fit[..half].copy_from_slice(&fitted[..half]);
        let start = self.number_of_points - wanted_window_size;
        x.copy_from_slice(&self.data_x[start..]);
        y.clear();
        for (pixel, index) in (start..self.number_of_points).enumerate() {
            y.push(if pixel > half {
                self.data_y[index]
            } else {
                self.savitzky_golay_fit[index]
            });
        }
        ls_poly(
            &x,
            &y,
            wanted_polynomial_order,
            &mut fitted,
            &mut self.savitzky_golay_coefficients[self.number_of_points - half],
        );
        for pixel in self.number_of_points - half + 1..self.number_of_points - 1 {
            self.savitzky_golay_coefficients[pixel] =
                self.savitzky_golay_coefficients[self.number_of_points - half].clone();
        }
        self.savitzky_golay_fit[self.number_of_points - half..]
            .copy_from_slice(&fitted[half + 1..]);
    }

    pub fn reciprocal(&mut self) {
        for value in &mut self.data_y {
            if *value != 0.0 {
                *value = 1.0 / *value;
            }
        }
    }

    pub fn allocate_savitzky_golay_coefficients(&mut self) {
        assert!(
            self.savitzky_golay_polynomial_order > 0,
            "Savitzky-Golay polynomial order was not set properly"
        );
        self.savitzky_golay_coefficients =
            vec![vec![0.0; self.savitzky_golay_polynomial_order + 1]; self.number_of_points];
    }
}

impl Default for Curve {
    fn default() -> Self {
        Self::new()
    }
}

/// C++ free function `LS_POLY`, using owned work vectors in place of C++ arrays.
pub fn ls_poly(
    x_data: &[f32],
    y_data: &[f32],
    order: usize,
    output_smoothed_curve: &mut [f32],
    output_coefficients: &mut [f32],
) {
    assert_eq!(x_data.len(), y_data.len());
    assert_eq!(x_data.len(), output_smoothed_curve.len());
    assert!(output_coefficients.len() > order);
    let n = x_data.len();
    assert!(
        n > order + 1,
        "polynomial order must leave degrees of freedom"
    );
    let mut a = vec![0.0_f64; order + 2];
    let mut b = vec![0.0_f64; order + 2];
    let mut c = vec![0.0_f64; order + 3];
    let mut c2 = vec![0.0_f64; order + 2];
    let mut f = vec![0.0_f64; order + 2];
    let mut v = vec![0.0_f64; n + 1];
    let mut d = vec![0.0_f64; n + 1];
    let mut e = vec![0.0_f64; n + 1];
    let mut x = vec![0.0_f64; n + 1];
    let mut y = vec![0.0_f64; n + 1];
    for index in 0..n {
        x[index + 1] = x_data[index] as f64;
        y[index + 1] = y_data[index] as f64;
    }
    let mut l = 0_usize;
    let n1 = order + 1;
    let mut v1 = 1.0e7_f64;
    for index in 1..=n1 {
        a[index] = 0.0;
        b[index] = 0.0;
        f[index] = 0.0;
    }
    let mut d1 = (n as f64).sqrt();
    let w = d1;
    for item in e.iter_mut().take(n + 1).skip(1) {
        *item = 1.0 / w;
    }
    let mut f1 = d1;
    let mut a1 = 0.0;
    for index in 1..=n {
        a1 += x[index] * e[index] * e[index];
    }
    let mut c1 = 0.0;
    for index in 1..=n {
        c1 += y[index] * e[index];
    }
    b[1] = 1.0 / f1;
    f[1] = b[1] * c1;
    for index in 1..=n {
        v[index] += e[index] * c1;
    }
    let mut m = 1_usize;
    let mut vv = 0.0;
    loop {
        if l > 0 {
            c2[1..=l].copy_from_slice(&c[1..=l]);
        }
        let l2 = l;
        let v2 = v1;
        let f2 = f1;
        let a2 = a1;
        f1 = 0.0;
        for index in 1..=n {
            let b1 = e[index];
            e[index] = (x[index] - a2) * e[index] - f2 * d[index];
            d[index] = b1;
            f1 += e[index] * e[index];
        }
        f1 = f1.sqrt();
        for item in e.iter_mut().take(n + 1).skip(1) {
            *item /= f1;
        }
        a1 = 0.0;
        for index in 1..=n {
            a1 += x[index] * e[index] * e[index];
        }
        c1 = 0.0;
        for index in 1..=n {
            c1 += e[index] * y[index];
        }
        m += 1;
        let mut index = 0;
        loop {
            l = m - index;
            let b2 = b[l];
            d1 = if l > 1 { b[l - 1] } else { 0.0 } - a2 * b[l] - f2 * a[l];
            b[l] = d1 / f1;
            a[l] = b2;
            index += 1;
            if index == m {
                break;
            }
        }
        for index in 1..=n {
            v[index] += e[index] * c1;
        }
        for index in 1..=n1 {
            f[index] += b[index] * c1;
            c[index] = f[index];
        }
        vv = (v
            .iter()
            .zip(&y)
            .skip(1)
            .map(|(fit, actual)| (fit - actual).powi(2))
            .sum::<f64>()
            / (n - l - 1) as f64)
            .sqrt();
        l = m;
        let e1 = 0.0_f64;
        if e1 == 0.0 {
            if m == n1 {
                break;
            }
            continue;
        }
        if (v1 - vv).abs() / vv < e1 || e1 * vv > e1 * v1 {
            l = l2;
            vv = v2;
            c[1..=l].copy_from_slice(&c2[1..=l]);
            break;
        }
        v1 = vv;
    }
    for index in 1..=l {
        c[index - 1] = c[index];
    }
    c[l] = 0.0;
    for index in 0..n {
        output_smoothed_curve[index] = v[index + 1] as f32;
    }
    for index in 0..=order {
        output_coefficients[index] = c[index] as f32;
    }
    let _ = vv;
}

#[cfg(test)]
mod tests {
    use super::Curve;

    #[test]
    fn weighted_addition_matches_bins() {
        let mut curve = Curve::new();
        curve.setup_x_axis(0.0, 2.0, 3);
        curve.data_y.copy_from_slice(&[0.0, 2.0, 4.0]);
        curve.add_value_at_x_using_linear_interpolation(0.5, 2.0, true);
        assert_eq!(curve.data_y, [1.0, 3.0, 4.0]);
    }
}
