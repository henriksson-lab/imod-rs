//! Translation of `IMOD/raptor/suitesparse/cs_scatter.c`.

use super::Cs;

/// C `cs_scatter`: accumulates one CSC column into a dense workspace and sparse pattern.
pub fn cs_scatter(
    matrix: &Cs,
    column: usize,
    beta: f64,
    marks: &mut [usize],
    workspace: Option<&mut [f64]>,
    mark: usize,
    pattern: &mut Vec<usize>,
) -> Option<usize> {
    if !matrix.is_csc() || column + 1 >= matrix.column_pointers.len() || marks.len() < matrix.rows {
        return None;
    }
    if let Some(values) = workspace.as_ref() {
        if values.len() < matrix.rows {
            return None;
        }
    }
    let mut workspace = workspace;
    for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
        let row = *matrix.row_indices.get(entry)?;
        let value = *matrix.values.get(entry)?;
        if row >= matrix.rows {
            return None;
        }
        if marks[row] < mark {
            marks[row] = mark;
            pattern.push(row);
            if let Some(values) = workspace.as_deref_mut() {
                values[row] = beta * value;
            }
        } else if let Some(values) = workspace.as_deref_mut() {
            values[row] += beta * value;
        }
    }
    Some(pattern.len())
}

#[cfg(test)]
mod tests {
    use super::cs_scatter;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn scatter_accumulates_repeated_rows_and_records_first_pattern_entry() {
        let matrix = Cs {
            nzmax: 3,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 3],
            row_indices: vec![0, 1, 0],
            values: vec![2., 3., 4.],
            nz: -1,
        };
        let mut marks = [0, 0];
        let mut values = [0., 0.];
        let mut pattern = vec![];
        assert_eq!(
            cs_scatter(
                &matrix,
                0,
                2.,
                &mut marks,
                Some(&mut values),
                1,
                &mut pattern
            ),
            Some(2)
        );
        assert_eq!(pattern, [0, 1]);
        assert_eq!(values, [12., 6.]);
    }
}
