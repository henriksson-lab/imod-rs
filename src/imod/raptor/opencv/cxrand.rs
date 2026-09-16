//! Pseudo-random filling from `IMOD/raptor/opencv/cxrand.cpp`.

use super::cxmean::{CvMeanMatrix, CvScalar};

/// C `CvRNG`, retaining the original 64-bit multiply-with-carry state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CvRng {
    pub state: u64,
}

impl CvRng {
    /// `cvRNG`: zero selects C's all-ones default seed.
    pub fn new(seed: i64) -> Self {
        Self {
            state: if seed == 0 { u64::MAX } else { seed as u64 },
        }
    }

    /// `cvRandInt`.
    pub fn rand_int(&mut self) -> u32 {
        self.state = (self.state as u32 as u64) * 1_554_115_554 + (self.state >> 32);
        self.state as u32
    }

    /// `cvRandReal`.
    pub fn rand_real(&mut self) -> f64 {
        self.rand_int() as f64 * 2.328_306_436_538_696_3e-10
    }
}

/// C `CV_RAND_UNI` and `CV_RAND_NORMAL`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvRandDistribution {
    Uniform,
    Normal,
}

/// Destination conversion used by C's `CV_CAST_*` macros.
pub trait CvRandValue: Copy {
    const INTEGER: bool;
    fn from_rand(value: f64) -> Self;
    fn uniform_unit(rng: &mut CvRng) -> f64;
}

macro_rules! impl_rand_integer {
    ($type:ty) => {
        impl CvRandValue for $type {
            const INTEGER: bool = true;
            fn from_rand(value: f64) -> Self {
                value
                    .round()
                    .clamp(<$type>::MIN as f64, <$type>::MAX as f64) as $type
            }
            fn uniform_unit(rng: &mut CvRng) -> f64 {
                f32::from_bits((rng.rand_int() >> 9) | 0x3f80_0000) as f64 - 1.0
            }
        }
    };
}

impl_rand_integer!(u8);
impl_rand_integer!(u16);
impl_rand_integer!(i16);
impl_rand_integer!(i32);

impl CvRandValue for f32 {
    const INTEGER: bool = false;
    fn from_rand(value: f64) -> Self {
        value as f32
    }
    fn uniform_unit(rng: &mut CvRng) -> f64 {
        f32::from_bits((rng.rand_int() >> 9) | 0x3f80_0000) as f64 - 1.0
    }
}

impl CvRandValue for f64 {
    const INTEGER: bool = false;
    fn from_rand(value: f64) -> Self {
        value
    }
    fn uniform_unit(rng: &mut CvRng) -> f64 {
        rng.rand_int();
        f64::from_bits(
            ((rng.state as u32 as u64) << 20) | (rng.state >> 44) | 0x3ff0_0000_0000_0000,
        ) - 1.0
    }
}

/// Errors from the safe `cvRandArr` translation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CvRandError {
    BadArgument,
    UnsupportedFormat,
}

/// `icvRandn_0_1_32f_C1R`, Marsaglia and Tsang's Monty Python generator.
pub fn icv_randn_0_1_32f_c1r(output: &mut [f32], rng: &mut CvRng) {
    let mut random = rng.rand_int();
    for value in output {
        let sample = loop {
            let mut x = random as i32 as f64 * 1.167_239e-9;
            let absolute_x = x.abs();
            let v = 2.8658 - absolute_x * (2.0213 - 0.3605 * absolute_x);
            let y = rng.rand_int() as f64 * 2.328_306e-10;
            random = rng.rand_int();
            if y < v || absolute_x < 1.17741 {
                break x;
            }
            let bx = x;
            x = if bx > 0.0 {
                0.8857913 * (2.506628 - absolute_x)
            } else {
                -0.8857913 * (2.506628 - absolute_x)
            };
            if y > v + 0.0506 {
                break x;
            }
            if y.ln() < 0.6931472 - 0.5 * bx * bx {
                break bx;
            }
            if (1.8857913 - y).ln() < 0.5718733 - 0.5 * x * x {
                break x;
            }
            let tail = loop {
                let tail_sign = random as i32 as f64 * 4.656_613e-10;
                let tail_x = -tail_sign.abs().ln() * 0.3989423;
                random = rng.rand_int();
                let tail_y = -(random as f64 * 2.328_306e-10).ln();
                random = rng.rand_int();
                if tail_y + tail_y >= tail_x * tail_x {
                    break (if tail_sign > 0.0 {
                        2.506628 + tail_x
                    } else {
                        -2.506628 - tail_x
                    });
                }
            };
            break tail;
        };
        *value = sample as f32;
    }
    // `random` is always the low word of the final source state.  The C loop
    // retains its high-word carry too, so the generator methods above have
    // already left `rng.state` at that exact state.
}

/// Owned generic `cvRandArr`.
pub fn cv_rand_arr<T: CvRandValue>(
    rng: &mut CvRng,
    matrix: &mut CvMeanMatrix<T>,
    distribution: CvRandDistribution,
    parameter1: CvScalar,
    parameter2: CvScalar,
) -> Result<(), CvRandError> {
    let elements = matrix
        .size
        .width
        .checked_mul(matrix.size.height)
        .and_then(|n| n.checked_mul(matrix.channels))
        .ok_or(CvRandError::BadArgument)?;
    if matrix.channels == 0 || matrix.channels > 4 || matrix.data.len() != elements {
        return Err(CvRandError::BadArgument);
    }
    match distribution {
        CvRandDistribution::Uniform => {
            for index in 0..elements {
                let channel = index % matrix.channels;
                let lower = parameter1.values[channel];
                let upper = parameter2.values[channel];
                let fast_range = upper.floor() - lower.ceil();
                let fast_mode =
                    T::INTEGER && fast_range > 0.0 && fast_range <= i32::MAX as f64 && {
                        let range = fast_range as u64;
                        range != 0 && (range & (range - 1)) == 0
                    };
                let value = if T::INTEGER {
                    let minimum = lower.ceil();
                    if fast_mode {
                        minimum + (rng.rand_int() & (fast_range as u32 - 1)) as f64
                    } else {
                        lower + T::uniform_unit(rng) * (upper - lower)
                    }
                } else {
                    lower + T::uniform_unit(rng) * (upper - lower)
                };
                matrix.data[index] = T::from_rand(if T::INTEGER && !fast_mode {
                    value.floor()
                } else {
                    value
                });
            }
        }
        CvRandDistribution::Normal => {
            let mut standard = vec![0.0_f32; elements];
            icv_randn_0_1_32f_c1r(&mut standard, rng);
            for index in 0..elements {
                let channel = index % matrix.channels;
                matrix.data[index] = T::from_rand(
                    standard[index] as f64 * parameter2.values[channel]
                        + parameter1.values[channel],
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::cvutils::CvSize;
    use super::*;

    #[test]
    fn rng_state_transition_and_real_scaling_match_cv_rand_int() {
        let mut rng = CvRng::new(1);
        assert_eq!(rng.rand_int(), 1_554_115_554);
        assert_eq!(rng.state, 1_554_115_554);
        let real = rng.rand_real();
        assert!((0.0..1.0).contains(&real));
    }

    #[test]
    fn uniform_integer_fast_path_stays_in_the_half_open_interval() {
        let mut rng = CvRng::new(7);
        let mut matrix = CvMeanMatrix {
            size: CvSize {
                width: 32,
                height: 1,
            },
            channels: 1,
            data: vec![0_u8; 32],
        };
        cv_rand_arr(
            &mut rng,
            &mut matrix,
            CvRandDistribution::Uniform,
            CvScalar { values: [10.0; 4] },
            CvScalar { values: [18.0; 4] },
        )
        .unwrap();
        assert!(matrix.data.iter().all(|&value| (10..18).contains(&value)));
    }

    #[test]
    fn normal_fill_is_reproducible_and_uses_channel_parameters() {
        let mut first_rng = CvRng::new(11);
        let mut second_rng = CvRng::new(11);
        let mut first = CvMeanMatrix {
            size: CvSize {
                width: 3,
                height: 1,
            },
            channels: 2,
            data: vec![0_f32; 6],
        };
        let mut second = first.clone();
        let mean = CvScalar {
            values: [2.0, -3.0, 0.0, 0.0],
        };
        let deviation = CvScalar {
            values: [0.0, 0.0, 0.0, 0.0],
        };
        cv_rand_arr(
            &mut first_rng,
            &mut first,
            CvRandDistribution::Normal,
            mean,
            deviation,
        )
        .unwrap();
        cv_rand_arr(
            &mut second_rng,
            &mut second,
            CvRandDistribution::Normal,
            mean,
            deviation,
        )
        .unwrap();
        assert_eq!(first.data, second.data);
        assert_eq!(first.data, [2.0, -3.0, 2.0, -3.0, 2.0, -3.0]);
    }
}
