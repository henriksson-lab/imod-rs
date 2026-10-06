//! Translation of `IMOD/flib/subrs/graphics/trnc.f90`.

use super::psplotpak::UNITS_PER_INCH;
use crate::imod::flib::subrs::compat::gfortran_rt::nint_r4;

/// Original `trnc` (`trnc.f90:1`): snaps a coordinate in inches to the
/// plotter's dot grid in groups of three dots, plus a safety offset.
///
/// Fixed in translation (BUGS.md, `trnc`): for a negative coordinate the
/// source's `mod(idot, 3)` is negative and `safe2(mod(idot, 3) + 1)` reads
/// before the array; here the dot is split with floor division and a
/// non-negative remainder, which is the source's arithmetic for every
/// coordinate that is not negative.
pub fn trnc(xx: f32) -> f32 {
    let safe2: [f32; 3] = [0.35, 1.2, 2.65];
    let units_per_inch = UNITS_PER_INCH.get();
    let idot = nint_r4(xx * units_per_inch);
    let group = idot.div_euclid(3);
    let rem = idot.rem_euclid(3);
    ((3 * group) as f32 + safe2[rem as usize]) / units_per_inch
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `BUGS.md` (`trnc`), defined behaviour: a negative coordinate snaps
    /// with floor division (no read before `safe2`); others are unchanged.
    #[test]
    fn negative_coordinates_snap_like_positive_ones() {
        // 300 units per inch (the module default)
        assert_eq!(trnc(0.01), (3. + 0.35) / 300.);
        assert_eq!(trnc(0.0167), (3. + 2.65) / 300.);
        // idot = -1: group -1, remainder 2
        assert_eq!(trnc(-0.004), (-3. + 2.65) / 300.);
        // idot = -3: group -1, remainder 0
        assert_eq!(trnc(-0.01), (-3. + 0.35) / 300.);
    }
}
