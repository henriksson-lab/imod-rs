//! Translation of `IMOD/libfft/realft.c`.

use super::cmplft;

/// C `realft`.
pub unsafe fn realft(even: *mut f32, odd: *mut f32, n: i32, dim: *mut i32) {
    unsafe {
        let twopi = 6.2831853_f32;
        let two_n = (2 * n) as f32;
        cmplft(even, odd, n, dim);
        let total = *dim.add(1);
        let along = *dim.add(2);
        let limit = *dim.add(3);
        let extent = *dim.add(4) - 1;
        let between = *dim.add(5);
        let over_two = n / 2 + 1;
        if over_two >= 2 {
            for index in 2..=over_two {
                let angle = twopi * (index - 1) as f32 / two_n;
                let co = (angle as f64).cos() as f32;
                let si = (angle as f64).sin() as f32;
                let first = (index - 1) * along + 1;
                let delta = (n + 2 - 2 * index) * along;
                let mut row = first;
                while row <= total {
                    let row_end = row + extent;
                    let mut point = row - 1;
                    while point < row_end {
                        let paired = point + delta;
                        let a = (*even.add(paired as usize) + *even.add(point as usize)) / 2.0;
                        let c = (*even.add(paired as usize) - *even.add(point as usize)) / 2.0;
                        let b = (*odd.add(paired as usize) + *odd.add(point as usize)) / 2.0;
                        let d = (*odd.add(paired as usize) - *odd.add(point as usize)) / 2.0;
                        let e = c * si + b * co;
                        let f = c * co - b * si;
                        *even.add(point as usize) = a + e;
                        *even.add(paired as usize) = a - e;
                        *odd.add(point as usize) = f - d;
                        *odd.add(paired as usize) = f + d;
                        point += between;
                    }
                    row += limit;
                }
            }
        }
        if n < 1 {
            return;
        }
        let delta = n * along;
        let mut row = 1;
        while row <= total {
            let row_end = row + extent;
            let mut point = row - 1;
            while point < row_end {
                let paired = point + delta;
                *even.add(paired as usize) = *even.add(point as usize) - *odd.add(point as usize);
                *odd.add(paired as usize) = 0.0;
                *even.add(point as usize) += *odd.add(point as usize);
                *odd.add(point as usize) = 0.0;
                point += between;
            }
            row += limit;
        }
    }
}
