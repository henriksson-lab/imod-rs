//! Translation of `IMOD/libfft/hermft.c`.

use super::cmplft;

/// C `hermft` over IMOD's interleaved real/imaginary float storage.
pub fn hermft(data: &mut [f32], n: i32, dim: &mut [i32; 6]) {
    assert!(data.len() >= dim[1] as usize);

    let twopi = 6.2831853_f32;
    let two_n = (2 * n) as f32;
    let total = dim[1];
    let along = dim[2];
    let limit = dim[3];
    let extent = dim[4] - 1;
    let between = dim[5];
    let mut first = 1;
    while first <= total {
        let end = first + extent;
        let mut index = first - 1;
        while index < end {
            let a = data[index as usize];
            let b = data[index as usize + 1];
            data[index as usize] = a + b;
            data[index as usize + 1] = a - b;
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
                let a = data[index as usize] + data[paired as usize];
                let b = data[index as usize] - data[paired as usize];
                let c = data[index as usize + 1] + data[paired as usize + 1];
                let d = data[index as usize + 1] - data[paired as usize + 1];
                let e = b * co + c * si;
                let f = b * si - c * co;
                data[index as usize] = a + f;
                data[paired as usize] = a - f;
                data[index as usize + 1] = e + d;
                data[paired as usize + 1] = e - d;
                index += between;
            }
            row += limit;
        }
    }
    cmplft(data, n, dim);
}

#[cfg(test)]
mod tests {
    use super::hermft;

    #[test]
    fn length_two_hermitian_pair_is_combined_in_interleaved_storage() {
        let mut values = [3.5_f32, -1.25];
        let mut dimensions = [0_i32, 2, 1, 2, 1, 2];

        hermft(&mut values, 1, &mut dimensions);

        assert_eq!(values, [2.25, 4.75]);
    }
}
