//! Mathematical functions from `IMOD/raptor/opencv/cxmathfuncs.cpp`.

/// C `ICV_MATH_BLOCK_SIZE`.
pub const ICV_MATH_BLOCK_SIZE: usize = 256;
/// Source fast-arctangent polynomial coefficients and quadrant table.
pub const ICV_ATAN_CF0: f32 = -15.813_189;
pub const ICV_ATAN_CF1: f32 = 61.094_193;
pub const ICV_ATAN_TAB: [f32; 8] = [0., 90., 180., 90., 360., 270., 180., 270.];

/// C `CV_CHECK_RANGE`.
pub const CV_CHECK_RANGE: u32 = 1;
/// C `CV_CHECK_QUIET`.
pub const CV_CHECK_QUIET: u32 = 2;

/// `cvFastArctan`, returning degrees in `[0, 360)`.
pub fn cv_fast_arctan(y: f32, x: f32) -> f32 {
    let angle = y.atan2(x).to_degrees();
    if angle < 0.0 { angle + 360.0 } else { angle }
}

/// `icvFastArctan_32f`.
pub fn icv_fast_arctan_32f(y: &[f32], x: &[f32], angle: &mut [f32]) -> Result<(), ()> {
    if y.len() != x.len() || angle.len() != x.len() {
        return Err(());
    }
    for index in 0..x.len() {
        angle[index] = cv_fast_arctan(y[index], x[index]);
    }
    Ok(())
}

/// `cvCbrt`.
pub fn cv_cbrt(value: f32) -> f32 {
    value.cbrt()
}

/// `icvInvSqrt_32f`.
pub fn icv_inv_sqrt_32f(source: &[f32], destination: &mut [f32]) -> Result<(), ()> {
    if source.len() != destination.len() {
        return Err(());
    }
    for index in 0..source.len() {
        destination[index] = 1.0 / source[index].sqrt();
    }
    Ok(())
}

/// `icvSqrt_32f`.
pub fn icv_sqrt_32f(source: &[f32], destination: &mut [f32]) -> Result<(), ()> {
    if source.len() != destination.len() {
        return Err(());
    }
    for index in 0..source.len() {
        destination[index] = source[index].sqrt();
    }
    Ok(())
}

/// `icvSqrt_64f`.
pub fn icv_sqrt_64f(source: &[f64], destination: &mut [f64]) -> Result<(), ()> {
    if source.len() != destination.len() {
        return Err(());
    }
    for index in 0..source.len() {
        destination[index] = source[index].sqrt();
    }
    Ok(())
}

/// Owned slice translation of `cvCartToPolar`.
pub fn cv_cart_to_polar(
    x: &[f64],
    y: &[f64],
    magnitude: &mut [f64],
    angle: Option<&mut [f64]>,
    angle_in_degrees: bool,
) -> Result<(), ()> {
    if x.len() != y.len()
        || magnitude.len() != x.len()
        || angle
            .as_ref()
            .map_or(false, |values| values.len() != x.len())
    {
        return Err(());
    }
    match angle {
        Some(angle) => {
            for index in 0..x.len() {
                magnitude[index] = x[index].hypot(y[index]);
                angle[index] = if angle_in_degrees {
                    y[index].atan2(x[index]).to_degrees()
                } else {
                    y[index].atan2(x[index])
                };
            }
        }
        None => {
            for index in 0..x.len() {
                magnitude[index] = x[index].hypot(y[index]);
            }
        }
    }
    Ok(())
}

/// Owned slice translation of `cvPolarToCart`.
pub fn cv_polar_to_cart(
    magnitude: Option<&[f64]>,
    angle: &[f64],
    x: &mut [f64],
    y: &mut [f64],
    angle_in_degrees: bool,
) -> Result<(), ()> {
    if x.len() != angle.len()
        || y.len() != angle.len()
        || magnitude.map_or(false, |values| values.len() != angle.len())
    {
        return Err(());
    }
    for index in 0..angle.len() {
        let radians = if angle_in_degrees {
            angle[index].to_radians()
        } else {
            angle[index]
        };
        let radius = magnitude.map_or(1.0, |values| values[index]);
        x[index] = radius * radians.cos();
        y[index] = radius * radians.sin();
    }
    Ok(())
}

/// `cvExp` for the numeric data path; Rust's correctly-rounded primitive
/// replaces the source's range-reduced approximation table.
pub fn cv_exp(source: &[f64], destination: &mut [f64]) -> Result<(), ()> {
    if source.len() != destination.len() {
        return Err(());
    }
    for index in 0..source.len() {
        destination[index] = source[index].exp();
    }
    Ok(())
}

/// `cvLog` for the numeric data path.
pub fn cv_log(source: &[f64], destination: &mut [f64]) -> Result<(), ()> {
    if source.len() != destination.len() {
        return Err(());
    }
    for index in 0..source.len() {
        destination[index] = source[index].ln();
    }
    Ok(())
}

/// `cvPow`.
pub fn cv_pow(source: &[f64], destination: &mut [f64], power: f64) -> Result<(), ()> {
    if source.len() != destination.len() {
        return Err(());
    }
    for index in 0..source.len() {
        destination[index] = source[index].powf(power);
    }
    Ok(())
}

/// `icvCheckArray_32f_C1R`/`icvCheckArray_64f_C1R` and `cvCheckArr`.
pub fn cv_check_arr(values: &[f64], flags: u32, minimum: f64, maximum: f64) -> bool {
    values.iter().all(|&value| {
        value.is_finite() && (flags & CV_CHECK_RANGE == 0 || (value >= minimum && value < maximum))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn angles_roots_and_polar_roundtrip_follow_source_contracts() {
        assert_eq!(cv_fast_arctan(0., -1.), 180.);
        assert!((cv_cbrt(-27.) + 3.).abs() < 1e-6);
        let mut magnitude = [0.; 2];
        let mut angle = [0.; 2];
        cv_cart_to_polar(
            &[3., 0.],
            &[4., 1.],
            &mut magnitude,
            Some(&mut angle),
            false,
        )
        .unwrap();
        let mut x = [0.; 2];
        let mut y = [0.; 2];
        cv_polar_to_cart(Some(&magnitude), &angle, &mut x, &mut y, false).unwrap();
        assert!((x[0] - 3.).abs() < 1e-12 && (y[0] - 4.).abs() < 1e-12);
    }
    #[test]
    fn exponential_logarithm_power_and_range_checks_work() {
        let mut values = [0.; 2];
        cv_exp(&[0., 1.], &mut values).unwrap();
        assert_eq!(values[0], 1.);
        cv_log(&values, &mut values.clone()).unwrap();
        let mut squared = [0.; 2];
        cv_pow(&[2., 3.], &mut squared, 2.).unwrap();
        assert_eq!(squared, [4., 9.]);
        assert!(cv_check_arr(&[0., 0.5], CV_CHECK_RANGE, 0., 1.));
        assert!(!cv_check_arr(&[f64::NAN], 0, 0., 1.));
    }
}
