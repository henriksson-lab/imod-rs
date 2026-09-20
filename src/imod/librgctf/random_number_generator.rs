//! Translation of `IMOD/librgctf/randomnumbergenerator.{h,cpp}`.

/// C++ `RandomNumberGenerator` (`randomnumbergenerator.h:3`).
///
/// `use_internal` selects the source's own linear congruential generator so a
/// program can hold several independent streams; the other mode is the
/// process-global C `rand`, which is what the source uses there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomNumberGenerator {
    pub random_seed: i32,
    pub use_internal: bool,
    pub next_seed: u32,
}

/// C's `RAND_MAX` on glibc.
const RAND_MAX: i32 = 2147483647;

impl RandomNumberGenerator {
    /// C++ `RandomNumberGenerator::RandomNumberGenerator(bool)`
    /// (`randomnumbergenerator.cpp:4`).
    pub fn new(internal: bool) -> Self {
        let mut this = Self {
            random_seed: 0,
            use_internal: internal,
            next_seed: 0,
        };
        this.use_internal = internal;
        this.set_seed(4711);
        this
    }

    /// C++ `RandomNumberGenerator::RandomNumberGenerator(int, bool)`
    /// (`randomnumbergenerator.cpp:10`).
    pub fn with_seed(random_seed: i32, internal: bool) -> Self {
        let mut this = Self {
            random_seed: 0,
            use_internal: internal,
            next_seed: 0,
        };
        this.use_internal = internal;
        this.set_seed(random_seed);
        this
    }

    /// C++ `RandomNumberGenerator::SetSeed` (`randomnumbergenerator.cpp:16`).
    pub fn set_seed(&mut self, random_seed: i32) {
        if random_seed < 0 {
            self.random_seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before Unix epoch")
                .as_secs() as i32;
        } else {
            self.random_seed = random_seed;
        }

        if self.use_internal {
            self.internal_srand(self.random_seed as u32);
        } else {
            // The source explicitly selects the process-global C stream here.
            unsafe { libc::srand(self.random_seed as u32) };
        }
    }

    /// C++ `RandomNumberGenerator::GetUniformRandom`
    /// (`randomnumbergenerator.cpp:30`): a uniform number in [-1,1].
    pub fn get_uniform_random(&mut self) -> f32 {
        let rnd1: f32;
        let hmax: f32;
        if self.use_internal {
            rnd1 = self.internal_rand() as f32;
            hmax = (32767.0 / 2.0f64) as f32;
        } else {
            rnd1 = (unsafe { libc::rand() }) as f32;
            hmax = (f64::from(RAND_MAX as f32) / 2.0) as f32;
        }
        (rnd1 - hmax) / hmax
    }

    /// C++ `RandomNumberGenerator::GetNormalRandom`
    /// (`randomnumbergenerator.cpp:52`): the polar Box-Muller transform.
    ///
    /// `sqrtf(-2.0 * log(R) / R)` is a `float` `logf` inside a **double**
    /// expression (`-2.0` is a double literal), and the double result is
    /// narrowed back to `float` by `sqrtf`'s parameter.
    pub fn get_normal_random(&mut self) -> f32 {
        let mut x1: f32;
        let mut x2: f32;
        let mut r: f32;
        loop {
            x1 = self.get_uniform_random();
            x2 = self.get_uniform_random();
            r = x1 * x1 + x2 * x2;
            if !(r == 0.0 || r > 1.0) {
                break;
            }
        }
        let _ = x2;
        x1 * ((-2.0 * f64::from(r.ln()) / f64::from(r)) as f32).sqrt()
    }

    /// C++ `RandomNumberGenerator::Internal_srand` (`randomnumbergenerator.cpp:67`).
    pub fn internal_srand(&mut self, random_seed: u32) {
        self.next_seed = random_seed;
    }

    /// C++ `RandomNumberGenerator::Internal_rand` (`randomnumbergenerator.cpp:72`).
    pub fn internal_rand(&mut self) -> i32 {
        self.next_seed = self.next_seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.next_seed / 65536) % 32768) as i32
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

        generator.set_seed(12);
        let first = generator.internal_rand();
        generator.set_seed(12);
        assert_eq!(generator.internal_rand(), first);
    }
}
