//! Translation of `IMOD/flib/subrs/hvem/getbinnedsize.f`.

/// Original `getBinnedSize` (`getbinnedsize.f:13`).
pub fn get_binned_size(nx: i32, nbin: i32, if_odd_even_ok: i32) -> (i32, i32) {
    let mut nxbin = nx / nbin;
    let mut irem = nx - nbin * nxbin;
    if nx.rem_euclid(2) == nxbin.rem_euclid(2) || (if_odd_even_ok != 0 && irem <= 1) {
        if irem > 1 {
            nxbin += 2;
        }
    } else {
        nxbin += 1;
        irem += nbin;
    }
    let mut ix_offset = 0;
    if irem > 1 {
        ix_offset = -(nbin - irem / 2);
    }
    (nxbin, ix_offset)
}

#[cfg(test)]
mod tests {
    use super::get_binned_size;

    #[test]
    fn source_even_odd_and_remainder_paths() {
        assert_eq!(get_binned_size(10, 3, 0), (4, -1));
        assert_eq!(get_binned_size(9, 4, 0), (3, -2));
        assert_eq!(get_binned_size(9, 4, 1), (2, 0));
        assert_eq!(get_binned_size(8, 4, 0), (2, 0));
    }
}
