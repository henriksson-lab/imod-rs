//! Translation of `IMOD/libfft/realft.c`.

use super::cmplft;

/// C `realft`.
pub fn realft(data: &mut [f32], n: i32, dim: &mut [i32; 6]) {
    assert!(data.len() >= dim[1] as usize);
    let twopi = 6.2831853_f32;
    let two_n = (2 * n) as f32;
    cmplft(data, n, dim);
    let total = dim[1];
    let along = dim[2];
    let limit = dim[3];
    let extent = dim[4] - 1;
    let between = dim[5];
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
                    let a = (data[paired as usize] + data[point as usize]) / 2.0;
                    let c = (data[paired as usize] - data[point as usize]) / 2.0;
                    let b = (data[paired as usize + 1] + data[point as usize + 1]) / 2.0;
                    let d = (data[paired as usize + 1] - data[point as usize + 1]) / 2.0;
                    let e = c * si + b * co;
                    let f = c * co - b * si;
                    data[point as usize] = a + e;
                    data[paired as usize] = a - e;
                    data[point as usize + 1] = f - d;
                    data[paired as usize + 1] = f + d;
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
            data[paired as usize] = data[point as usize] - data[point as usize + 1];
            data[paired as usize + 1] = 0.0;
            data[point as usize] += data[point as usize + 1];
            data[point as usize + 1] = 0.0;
            point += between;
        }
        row += limit;
    }
}
