//! Translation of `IMOD/librgctf/randomnumbergenerator.{h,cpp}`.

/// The source-compatible random-number state used by librgctf.
///
/// `use_internal` selects IMOD's local ANSI-C linear congruential generator,
/// allowing independent streams.  The other mode deliberately retains the
/// source's process-global C `rand` stream for parity with existing callers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RandomNumberGenerator {
    random_seed: i32,
    use_internal: bool,
    next_seed: u32,
}

impl RandomNumberGenerator {
    /// C++ `RandomNumberGenerator::RandomNumberGenerator(bool)`.
    pub fn new(use_internal: bool) -> Self {
        let mut generator = Self {
            random_seed: 0,
            use_internal,
            next_seed: 0,
        };
        generator.set_seed(4711);
        generator
    }

    /// C++ `RandomNumberGenerator::RandomNumberGenerator(int, bool)`.
    pub fn with_seed(random_seed: i32, use_internal: bool) -> Self {
        let mut generator = Self {
            random_seed: 0,
            use_internal,
            next_seed: 0,
        };
        generator.set_seed(random_seed);
        generator
    }

    /// C++ `RandomNumberGenerator::SetSeed`.
    pub fn set_seed(&mut self, random_seed: i32) {
        self.random_seed = if random_seed < 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before Unix epoch")
                .as_secs() as i32
        } else {
            random_seed
        };

        if self.use_internal {
            self.internal_srand(self.random_seed as u32);
        } else {
            // The C++ source explicitly selects the process-global C stream
            // in this mode.  No C value crosses this Rust API boundary.
            unsafe { libc::srand(self.random_seed as u32) };
        }
    }

    /// C++ `RandomNumberGenerator::GetUniformRandom`.
    pub fn uniform_random(&mut self) -> f32 {
        let (random, half_maximum) = if self.use_internal {
            (self.internal_rand() as f32, 32767.0 / 2.0)
        } else {
            (unsafe { libc::rand() } as f32, libc::RAND_MAX as f32 / 2.0)
        };
        (random - half_maximum) / half_maximum
    }

    /// C++ `RandomNumberGenerator::GetNormalRandom`.
    pub fn normal_random(&mut self) -> f32 {
        let (mut x1, mut x2, mut radius_squared);
        loop {
            x1 = self.uniform_random();
            x2 = self.uniform_random();
            radius_squared = x1 * x1 + x2 * x2;
            if radius_squared != 0.0 && radius_squared <= 1.0 {
                break;
            }
        }
        x1 * (-2.0 * radius_squared.ln() / radius_squared).sqrt()
    }

    /// C++ `RandomNumberGenerator::Internal_srand`.
    pub fn internal_srand(&mut self, random_seed: u32) {
        self.next_seed = random_seed;
    }

    /// C++ `RandomNumberGenerator::Internal_rand`.
    pub fn internal_rand(&mut self) -> i32 {
        self.next_seed = self
            .next_seed
            .wrapping_mul(1_103_515_245)
            .wrapping_add(12_345);
        ((self.next_seed / 65_536) % 32_768) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::RandomNumberGenerator;

    #[test]
    fn internal_generator_matches_source_lcg() {
        let mut generator = RandomNumberGenerator::new(true);
        assert_eq!(generator.internal_rand(), 26_701);
        assert_eq!(generator.internal_rand(), 4_643);

        generator.set_seed(4711);
        assert!((generator.uniform_random() - 0.629_749_4).abs() < 0.000_001);
    }

    #[test]
    fn explicit_seed_restarts_internal_stream() {
        let mut generator = RandomNumberGenerator::with_seed(12, true);
        let first = generator.internal_rand();
        generator.set_seed(12);
        assert_eq!(generator.internal_rand(), first);
    }
}
