//! Translation of `IMOD/libiimod/halffloat.c`.

pub fn imnp_halfbuf_to_floats(half_buf: &[u16], float_buf: &mut [f32], num_vals: i32) {
    for ind in 0..num_vals as usize {
        float_buf[ind] = f32::from_bits(imnp_halfbits_to_floatbits(half_buf[ind]));
    }
}
pub fn imnp_floatbuf_to_halfs(float_buf: &[f32], half_buf: &mut [u16], num_vals: i32) {
    for ind in 0..num_vals as usize {
        half_buf[ind] = imnp_floatbits_to_halfbits(float_buf[ind].to_bits());
    }
}
pub fn imnp_floatbits_to_halfbits(f: u32) -> u16 {
    let f_exp = f & 0x7f80_0000;
    let h_sgn = ((f & 0x8000_0000) >> 16) as u16;
    if f_exp >= 0x4780_0000 {
        return h_sgn + 0x7bff;
    }
    if f_exp <= 0x3800_0000 {
        if f_exp < 0x3300_0000 {
            return h_sgn;
        }
        let mut f_sig = 0x0080_0000 + (f & 0x007f_ffff);
        f_sig >>= 113 - (f_exp >> 23);
        if (f_sig & 0x3fff) != 0x1000 || (f & 0x7ff) != 0 {
            f_sig += 0x1000;
        }
        return h_sgn + (f_sig >> 13) as u16;
    }
    let mut f_sig = f & 0x007f_ffff;
    if (f_sig & 0x3fff) != 0x1000 {
        f_sig += 0x1000;
    }
    let h_sig = ((f_sig >> 13) as u16) + ((f_exp - 0x3800_0000) >> 13) as u16;
    if h_sig == 0x7c00 {
        h_sgn + 0x7bff
    } else {
        h_sgn + h_sig
    }
}
pub fn imnp_halfbits_to_floatbits(h: u16) -> u32 {
    let mut h_exp = h & 0x7c00;
    let f_sgn = ((h as u32) & 0x8000) << 16;
    match h_exp {
        0 => {
            let mut h_sig = h & 0x03ff;
            if h_sig == 0 {
                return f_sgn;
            }
            h_sig <<= 1;
            while h_sig & 0x0400 == 0 {
                h_sig <<= 1;
                h_exp += 1;
            }
            f_sgn + (((127 - 15 - h_exp as i32) as u32) << 23) + ((h_sig as u32 & 0x03ff) << 13)
        }
        0x7c00 => f_sgn + 0x7f80_0000 + (((h & 0x03ff) as u32) << 13),
        _ => f_sgn + (((h as u32 & 0x7fff) + 0x1c000) << 13),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_bits_match_source_limits_and_round_trip() {
        assert_eq!(imnp_floatbits_to_halfbits(f32::INFINITY.to_bits()), 0x7bff);
        assert_eq!(imnp_halfbits_to_floatbits(0x3c00), 1.0f32.to_bits());
        let in_f = [-2.0, 0.0, 1.5];
        let mut halves = [0; 3];
        let mut out = [0.; 3];
        imnp_floatbuf_to_halfs(&in_f, &mut halves, 3);
        imnp_halfbuf_to_floats(&halves, &mut out, 3);
        assert_eq!(in_f, out);
    }
}
