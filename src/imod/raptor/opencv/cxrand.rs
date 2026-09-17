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

/// `icvRand_64f_C1R` (`cxrand.cpp:211`).  `parameters` is the source's
/// 24-element table: twelve repeated offsets followed by twelve scales.  The
/// explicit width/height/stride form retains its row-padding behavior while
/// replacing raw pointers with checked slices.
pub fn icv_rand_64f_c1r(
    output: &mut [f64],
    row_stride: usize,
    width: usize,
    height: usize,
    rng: &mut CvRng,
    parameters: &[f64],
) -> Result<(), CvRandError> {
    if parameters.len() < 24
        || row_stride < width
        || (height > 0 && output.len() < (height - 1) * row_stride + width)
    {
        return Err(CvRandError::BadArgument);
    }
    for row in 0..height {
        let mut parameter_base = 0;
        let mut groups_left = 3;
        for column in 0..width {
            let slot = column % 4;
            output[row * row_stride + column] = <f64 as CvRandValue>::uniform_unit(rng)
                * parameters[parameter_base + slot + 12]
                + parameters[parameter_base + slot];
            if slot == 3 {
                parameter_base += 4;
                groups_left -= 1;
                if groups_left == 0 {
                    parameter_base = 0;
                    groups_left = 3;
                }
            }
        }
    }
    Ok(())
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
            let mut integer_minima = [0.0; 4];
            let mut integer_masks = [0_u32; 4];
            let mut fast_mode = T::INTEGER;
            for channel in 0..matrix.channels {
                let lower = parameter1.values[channel];
                let range = parameter2.values[channel].floor() - lower.ceil();
                let range_is_power_of_two =
                    range > 0.0 && range <= i32::MAX as f64 && (range as u32).is_power_of_two();
                fast_mode &= range_is_power_of_two;
                integer_minima[channel] = lower.ceil();
                integer_masks[channel] = range as u32 - 1;
            }

            // `icvRandBits_*`: when every mask fits in a byte, native OpenCV
            // uses the four byte lanes of one RNG word for four consecutive
            // output scalars.  This affects both the samples and `CvRNG`'s
            // observable final state.
            if fast_mode {
                let small_masks = integer_masks[..matrix.channels]
                    .iter()
                    .all(|&mask| mask <= 255);
                let mut index = 0;
                while index < elements {
                    let random = rng.rand_int();
                    let group = if small_masks && elements - index >= 4 {
                        4
                    } else {
                        1
                    };
                    for lane in 0..group {
                        if index == elements {
                            break;
                        }
                        let channel = index % matrix.channels;
                        let bits = if small_masks {
                            random >> (lane * 8)
                        } else {
                            random
                        };
                        matrix.data[index] = T::from_rand(
                            integer_minima[channel] + (bits & integer_masks[channel]) as f64,
                        );
                        index += 1;
                    }
                }
            } else {
                for index in 0..elements {
                    let channel = index % matrix.channels;
                    let lower = parameter1.values[channel];
                    let upper = parameter2.values[channel];
                    let value = lower + T::uniform_unit(rng) * (upper - lower);
                    matrix.data[index] =
                        T::from_rand(if T::INTEGER { value.floor() } else { value });
                }
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
        let mut expected_rng = rng;
        let expected_word = expected_rng.rand_int();
        for _ in 1..8 {
            expected_rng.rand_int();
        }
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
        assert_eq!(rng.state, expected_rng.state);
        assert_eq!(
            matrix.data[..4],
            [
                10 + (expected_word & 7) as u8,
                10 + ((expected_word >> 8) & 7) as u8,
                10 + ((expected_word >> 16) & 7) as u8,
                10 + ((expected_word >> 24) & 7) as u8,
            ]
        );
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

    #[test]
    fn native_f64_random_fill_preserves_row_padding() {
        let mut rng = CvRng::new(13);
        let mut output = [-1.0; 8];
        let mut parameters = [0.0; 24];
        parameters[..12].fill(2.0);
        parameters[12..].fill(0.0);
        icv_rand_64f_c1r(&mut output, 4, 3, 2, &mut rng, &parameters).unwrap();
        assert_eq!(output, [2.0, 2.0, 2.0, -1.0, 2.0, 2.0, 2.0, -1.0]);
    }
}
