use imod_core::MrcMode;

/// A 2D image slice with pixel data stored as f32.
///
/// This replaces IMOD's `Islice` struct which uses a union of typed pointers.
/// We normalize all data to f32 for processing, converting on read/write.
#[derive(Debug, Clone)]
pub struct Slice {
    pub nx: usize,
    pub ny: usize,
    pub data: Vec<f32>,
    /// Original data mode (for tracking source format).
    pub mode: MrcMode,
}

impl Slice {
    /// Create a new slice filled with a constant value.
    pub fn new(nx: usize, ny: usize, fill: f32) -> Self {
        Self {
            nx,
            ny,
            data: vec![fill; nx * ny],
            mode: MrcMode::Float,
        }
    }

    /// Create from existing f32 data.
    pub fn from_data(nx: usize, ny: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), nx * ny);
        Self {
            nx,
            ny,
            data,
            mode: MrcMode::Float,
        }
    }

    /// Number of pixels.
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }

    /// Returns `true` if the slice contains no pixels.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get pixel value at (x, y).
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.nx + x]
    }

    /// Set pixel value at (x, y).
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, val: f32) {
        self.data[y * self.nx + x] = val;
    }

    /// Get pixel with bounds checking, returning fill value if out of bounds.
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize, fill: f32) -> f32 {
        if x < 0 || y < 0 || x >= self.nx as isize || y >= self.ny as isize {
            fill
        } else {
            self.data[y as usize * self.nx + x as usize]
        }
    }

    /// Bilinear interpolation at fractional coordinates.
    pub fn interpolate_bilinear(&self, x: f32, y: f32, fill: f32) -> f32 {
        let x0 = x.floor() as isize;
        let y0 = y.floor() as isize;
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let v00 = self.get_clamped(x0, y0, fill);
        let v10 = self.get_clamped(x0 + 1, y0, fill);
        let v01 = self.get_clamped(x0, y0 + 1, fill);
        let v11 = self.get_clamped(x0 + 1, y0 + 1, fill);

        v00 * (1.0 - fx) * (1.0 - fy)
            + v10 * fx * (1.0 - fy)
            + v01 * (1.0 - fx) * fy
            + v11 * fx * fy
    }

    /// Min, max, mean of the slice.
    pub fn statistics(&self) -> (f32, f32, f32) {
        imod_core::Point3f::default(); // just to use imod_core
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum = 0.0_f64;
        for &v in &self.data {
            if v < min { min = v; }
            if v > max { max = v; }
            sum += v as f64;
        }
        (min, max, (sum / self.data.len() as f64) as f32)
    }

    /// Extract a subregion as a new slice.
    pub fn subregion(&self, x0: usize, y0: usize, w: usize, h: usize) -> Slice {
        let mut data = Vec::with_capacity(w * h);
        for y in y0..y0 + h {
            let start = y * self.nx + x0;
            data.extend_from_slice(&self.data[start..start + w]);
        }
        Slice {
            nx: w,
            ny: h,
            data,
            mode: self.mode,
        }
    }
}
