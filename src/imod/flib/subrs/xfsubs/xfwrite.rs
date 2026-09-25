//! Translation of `IMOD/flib/subrs/xfsubs/xfwrite.f`.

use std::io::Write;

/// Original `xfwrite` (`xfwrite.f:2`).
pub fn xfwrite<W: Write>(iunit: &mut W, f: &[f32; 6]) -> Result<(), ()> {
    // `101 format(4f12.7,2f12.3)` with source list-directed matrix order.
    // gfortran's `Fw.d` editing (`libgfortran/io/write_float.def`): a value
    // whose digits do not fit the field is written as `w` asterisks, and a
    // NaN or infinity is `NaN`/`Infinity`/`-Infinity` right-justified.
    // Rounding agrees with Rust's `{:.d}` (both round the exact binary value,
    // ties to even), measured against gfortran on tie values.
    let field = |value: f32, digits: usize| -> String {
        if value.is_nan() {
            return format!("{:>12}", "NaN");
        }
        if value.is_infinite() {
            return format!("{:>12}", if value < 0. { "-Infinity" } else { "Infinity" });
        }
        let text = format!("{value:12.digits$}");
        if text.len() > 12 {
            "*".repeat(12)
        } else {
            text
        }
    };
    writeln!(
        iunit,
        "{}{}{}{}{}{}",
        field(f[0], 7),
        field(f[2], 7),
        field(f[1], 7),
        field(f[3], 7),
        field(f[4], 3),
        field(f[5], 3)
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
