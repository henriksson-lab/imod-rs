//! Translation of `IMOD/flib/subrs/xfsubs/xfwrite.f`.

use std::io::Write;

/// Original `xfwrite` (`xfwrite.f:2`).
pub fn xfwrite<W: Write>(iunit: &mut W, f: &[f32; 6]) -> Result<(), ()> {
    // `101 format(4f12.7,2f12.3)` with source list-directed matrix order.
    writeln!(
        iunit,
        "{:12.7}{:12.7}{:12.7}{:12.7}{:12.3}{:12.3}",
        f[0], f[2], f[1], f[3], f[4], f[5]
    )
    .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::xfwrite;

    #[test]
    fn preserves_source_format_and_fortran_matrix_order() {
        let mut output = Vec::new();
        xfwrite(&mut output, &[1., 3., 2., 4., 5., 6.]).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "   1.0000000   2.0000000   3.0000000   4.0000000       5.000       6.000\n"
        );
    }
}
