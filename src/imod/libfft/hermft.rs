//! Translation of `IMOD/libfft/hermft.c`.
use super::cmplft;

/// C `hermft`.
pub unsafe fn hermft(x: *mut f32, y: *mut f32, n: i32, dim: *mut i32) {
    unsafe {
        let twopi = 6.2831853_f32;
        let two_n = (2 * n) as f32;
        let total = *dim.add(1);
        let along = *dim.add(2);
        let limit = *dim.add(3);
        let extent = *dim.add(4) - 1;
        let between = *dim.add(5);
        let mut first = 1;
        while first <= total {
            let end = first + extent;
            let mut index = first - 1;
            while index < end {
                let a = *x.add(index as usize);
                let b = *y.add(index as usize);
                *x.add(index as usize) = a + b;
                *y.add(index as usize) = a - b;
                index += between;
            }
            first += limit;
        }
        let over_two = n / 2 + 1;
        if over_two < 2 {
            return;
        }
        for source in 2..=over_two {
            let angle = twopi * (source - 1) as f32 / two_n;
            let co = (angle as f64).cos() as f32;
            let si = (angle as f64).sin() as f32;
            let delta = (n + 2 - 2 * source) * along;
            let mut row = (source - 1) * along + 1;
            while row <= total {
                let end = row + extent;
                let mut index = row - 1;
                while index < end {
                    let paired = index + delta;
                    let a = *x.add(index as usize) + *x.add(paired as usize);
                    let b = *x.add(index as usize) - *x.add(paired as usize);
                    let c = *y.add(index as usize) + *y.add(paired as usize);
                    let d = *y.add(index as usize) - *y.add(paired as usize);
                    let e = b * co + c * si;
                    let f = b * si - c * co;
                    *x.add(index as usize) = a + f;
                    *x.add(paired as usize) = a - f;
                    *y.add(index as usize) = e + d;
                    *y.add(paired as usize) = e - d;
                    index += between;
                }
                row += limit;
            }
        }
        cmplft(x, y, n, dim);
    }
}
