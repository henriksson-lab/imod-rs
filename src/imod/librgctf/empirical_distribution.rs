//! Translation of `IMOD/librgctf/empirical_distribution.{h,cpp}`.

/// C++ `EmpiricalDistribution` (`empirical_distribution.h:1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmpiricalDistribution {
    sum_of_samples: f64,
    sum_of_squared_samples: f64,
    number_of_samples: i64,
    minimum: f32,
    maximum: f32,
    is_constant: bool,
    last_added_value: f32,
}

impl Default for EmpiricalDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl EmpiricalDistribution {
    /// C++ `EmpiricalDistribution::EmpiricalDistribution` (`empirical_distribution.cpp:4`).
    pub fn new() -> Self {
        let mut this = Self {
            sum_of_samples: 0.0,
            sum_of_squared_samples: 0.0,
            number_of_samples: 0,
            minimum: 0.0,
            maximum: 0.0,
            is_constant: false,
            last_added_value: 0.0,
        };
        this.reset();
        this
    }

    /// C++ `EmpiricalDistribution::Reset` (`empirical_distribution.cpp:15`).
    pub fn reset(&mut self) {
        self.sum_of_samples = 0.0;
        self.sum_of_squared_samples = 0.0;
        self.number_of_samples = 0;
        self.minimum = f32::MAX;
        self.maximum = -f32::MAX;
        self.is_constant = true;
        self.last_added_value = 0.0;
    }

    /// C++ `EmpiricalDistribution::AddSampleValue` (`empirical_distribution.cpp:26`).
    ///
    /// `pow(sample_value, 2)` selects the `double` overload, so the square is
    /// accumulated in double precision.  `std::min`/`std::max` are written
    /// `(b < a) ? b : a` and `(a < b) ? b : a`, which for a NaN sample return
    /// the running value — the same answer as Rust's `f32::min`/`max` here.
    pub fn add_sample_value(&mut self, sample_value: f32) {
        self.sum_of_samples += f64::from(sample_value);
        self.sum_of_squared_samples += f64::from(sample_value).powi(2);
        self.number_of_samples += 1;
        self.minimum = if sample_value < self.minimum {
            sample_value
        } else {
            self.minimum
        };
        self.maximum = if self.maximum < sample_value {
            sample_value
        } else {
            self.maximum
        };
        if self.number_of_samples == 1 {
            self.is_constant = true;
        } else {
            self.is_constant = self.is_constant && self.last_added_value == sample_value;
        }
        self.last_added_value = sample_value;
    }

    /// C++ `EmpiricalDistribution::IsConstant` (`empirical_distribution.cpp:44`).
    pub fn is_constant(&self) -> bool {
        self.is_constant
    }

    /// C++ `EmpiricalDistribution::GetSampleSumOfSquares` (`empirical_distribution.cpp:49`).
    pub fn get_sample_sum_of_squares(&self) -> f32 {
        self.sum_of_squared_samples as f32
    }

    /// C++ `EmpiricalDistribution::GetNumberOfSamples` (`empirical_distribution.cpp:54`).
    pub fn get_number_of_samples(&self) -> f32 {
        self.number_of_samples as f32
    }

    /// C++ `EmpiricalDistribution::GetSampleSum` (`empirical_distribution.cpp:59`).
    pub fn get_sample_sum(&self) -> f32 {
        self.sum_of_samples as f32
    }

    /// C++ `EmpiricalDistribution::GetSampleMean` (`empirical_distribution.cpp:64`).
    pub fn get_sample_mean(&self) -> f32 {
        if self.number_of_samples > 0 {
            (self.sum_of_samples / f64::from(self.number_of_samples as f32)) as f32
        } else {
            0.0
        }
    }

    /// C++ `EmpiricalDistribution::GetSampleVariance` (`empirical_distribution.cpp:76`).
    pub fn get_sample_variance(&self) -> f32 {
        if self.number_of_samples > 0 {
            ((self.sum_of_squared_samples / f64::from(self.number_of_samples as f32))
                - (self.sum_of_samples / f64::from(self.number_of_samples as f32)).powi(2))
                as f32
        } else {
            0.0
        }
    }

    /// C++ `EmpiricalDistribution::GetUnbiasedEstimateOfPopulationVariance`
    /// (`empirical_distribution.cpp:88`).
    pub fn get_unbiased_estimate_of_population_variance(&self) -> f32 {
        if self.number_of_samples > 0 {
            self.get_sample_variance() * (self.number_of_samples as f32)
                / ((self.number_of_samples - 1) as f32)
        } else {
            0.0
        }
    }

    /// C++ inline `EmpiricalDistribution::GetMinimum` (`empirical_distribution.h:26`).
    pub fn get_minimum(&self) -> f32 {
        self.minimum
    }

    /// C++ inline `EmpiricalDistribution::GetMaximum` (`empirical_distribution.h:27`).
    pub fn get_maximum(&self) -> f32 {
        self.maximum
    }
}

#[cfg(test)]
mod tests {
    use super::EmpiricalDistribution;

    #[test]
    fn tracks_source_statistics_and_reset_state() {
        let mut distribution = EmpiricalDistribution::new();
        assert!(distribution.is_constant());
        assert_eq!(distribution.get_minimum(), f32::MAX);
        assert_eq!(distribution.get_maximum(), -f32::MAX);

        distribution.add_sample_value(2.0);
        distribution.add_sample_value(4.0);
        distribution.add_sample_value(6.0);
        assert_eq!(distribution.get_sample_sum(), 12.0);
        assert_eq!(distribution.get_number_of_samples(), 3.0);
        assert_eq!(distribution.get_sample_sum_of_squares(), 56.0);
        assert_eq!(distribution.get_sample_mean(), 4.0);
        assert_eq!(distribution.get_sample_variance(), 8.0 / 3.0);
        assert_eq!(distribution.get_minimum(), 2.0);
        assert_eq!(distribution.get_maximum(), 6.0);
        assert!(!distribution.is_constant());

        distribution.reset();
        assert!(distribution.is_constant());
    }
}
