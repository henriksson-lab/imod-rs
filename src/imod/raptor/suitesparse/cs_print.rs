//! Translation of `IMOD/raptor/suitesparse/cs_print.c`.

use std::fmt::Write;

use super::Cs;

/// C `cs_print`: renders a CSparse matrix using its source diagnostic layout.
///
/// The C routine writes to `stdout`; this safe Rust form returns the owned
/// rendering instead.  An error carries C's `(null)` rendering for a missing
/// matrix, or describes malformed owned CSC/triplet storage.
pub fn cs_print(matrix: Option<&Cs>, brief: bool) -> Result<String, String> {
    let Some(matrix) = matrix else {
        return Err("(null)\n".to_owned());
    };
    let c_g = |value: f64| {
        if value.is_nan() {
            return "nan".to_owned();
        }
        if value.is_infinite() {
            return if value.is_sign_negative() {
                "-inf"
            } else {
                "inf"
            }
            .to_owned();
        }
        if value == 0.0 {
            return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
        }
        // `%g` chooses notation after rounding to six significant digits.
        // The exponent carried by Rust's five-decimal scientific rendering is
        // therefore the matching post-rounding exponent.
        let scientific = format!("{value:.5e}");
        let Some((scientific_mantissa, scientific_exponent)) = scientific.split_once('e') else {
            return scientific;
        };
        let Ok(exponent) = scientific_exponent.parse::<i32>() else {
            return scientific;
        };
        let mut rendered = if !(-4..6).contains(&exponent) {
            let mantissa = scientific_mantissa
                .trim_end_matches('0')
                .trim_end_matches('.');
            format!("{mantissa}e{exponent:+03}")
        } else {
            let decimals = (5 - exponent).max(0) as usize;
            format!("{value:.decimals$}")
        };
        if let Some(decimal) = rendered.find('.') {
            let exponent_at = rendered.find('e').unwrap_or(rendered.len());
            let trimmed = rendered[decimal..exponent_at]
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned();
            rendered.replace_range(decimal..exponent_at, &trimmed);
        }
        rendered
    };
    let values_present = !matrix.values.is_empty();
    let value_at = |entry: usize| {
        if values_present {
            matrix.values.get(entry).copied()
        } else {
            Some(1.0)
        }
    };
    let mut output = String::new();
    writeln!(
        output,
        "CSparse Version 2.2.3, Jan 20, 2009.  Copyright (c) Timothy A. Davis, 2006-2009"
    )
    .expect("writing to String cannot fail");

    if matrix.is_csc() {
        if matrix.column_pointers.len() < matrix.columns + 1
            || matrix
                .column_pointers
                .windows(2)
                .any(|pair| pair[0] > pair[1])
        {
            return Err("invalid CSC column pointers".to_owned());
        }
        let entries = matrix.column_pointers[matrix.columns];
        if entries > matrix.row_indices.len() || values_present && entries > matrix.values.len() {
            return Err("invalid CSC entry storage".to_owned());
        }
        let mut norm = 0.0_f64;
        if values_present {
            for column in 0..matrix.columns {
                let sum: f64 = (matrix.column_pointers[column]..matrix.column_pointers[column + 1])
                    .map(|entry| matrix.values[entry].abs())
                    .sum();
                norm = norm.max(sum);
            }
        } else {
            norm = -1.0;
        }
        writeln!(
            output,
            "{}-by-{}, nzmax: {} nnz: {}, 1-norm: {}",
            matrix.rows,
            matrix.columns,
            matrix.nzmax,
            entries,
            c_g(norm),
        )
        .expect("writing to String cannot fail");
        for column in 0..matrix.columns {
            writeln!(
                output,
                "    col {column} : locations {} to {}",
                matrix.column_pointers[column],
                matrix.column_pointers[column + 1] as i128 - 1,
            )
            .expect("writing to String cannot fail");
            for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
                writeln!(
                    output,
                    "      {} : {}",
                    matrix.row_indices[entry],
                    c_g(value_at(entry).ok_or_else(|| "invalid CSC values".to_owned())?),
                )
                .expect("writing to String cannot fail");
                if brief && entry > 20 {
                    output.push_str("  ...\n");
                    return Ok(output);
                }
            }
        }
    } else {
        let entries = usize::try_from(matrix.nz).map_err(|_| "invalid triplet count".to_owned())?;
        if entries > matrix.column_pointers.len()
            || entries > matrix.row_indices.len()
            || values_present && entries > matrix.values.len()
        {
            return Err("invalid triplet entry storage".to_owned());
        }
        writeln!(
            output,
            "triplet: {}-by-{}, nzmax: {} nnz: {entries}",
            matrix.rows, matrix.columns, matrix.nzmax,
        )
        .expect("writing to String cannot fail");
        for entry in 0..entries {
            writeln!(
                output,
                "    {} {} : {}",
                matrix.row_indices[entry],
                matrix.column_pointers[entry],
                c_g(value_at(entry).ok_or_else(|| "invalid triplet values".to_owned())?),
            )
            .expect("writing to String cannot fail");
            if brief && entry > 20 {
                output.push_str("  ...\n");
                return Ok(output);
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::cs_print;
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn print_renders_csc_header_columns_and_c_g_numbers() {
        let matrix = Cs {
            nzmax: 3,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 2, 3],
            row_indices: vec![0, 1, 1],
            values: vec![1.25, -2.0, 1_000_000.0],
            nz: -1,
        };
        assert_eq!(
            cs_print(Some(&matrix), false),
            Ok(concat!(
                "CSparse Version 2.2.3, Jan 20, 2009.  Copyright (c) Timothy A. Davis, 2006-2009\n",
                "2-by-2, nzmax: 3 nnz: 3, 1-norm: 1e+06\n",
                "    col 0 : locations 0 to 1\n",
                "      0 : 1.25\n",
                "      1 : -2\n",
                "    col 1 : locations 2 to 2\n",
                "      1 : 1e+06\n",
            )
            .to_owned())
        );
    }

    #[test]
    fn print_handles_triplets_brief_mode_and_null_source_output() {
        let triplet = Cs {
            nzmax: 22,
            rows: 2,
            columns: 2,
            column_pointers: (0..22).map(|value| value % 2).collect(),
            row_indices: (0..22).map(|value| value % 2).collect(),
            values: vec![],
            nz: 22,
        };
        let output = cs_print(Some(&triplet), true).unwrap();
        assert!(output.contains("triplet: 2-by-2, nzmax: 22 nnz: 22\n"));
        assert!(output.ends_with("    1 1 : 1\n  ...\n"));
        assert_eq!(cs_print(None, false), Err("(null)\n".to_owned()));
    }
}
