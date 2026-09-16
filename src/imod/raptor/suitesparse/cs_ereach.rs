//! Translation of `IMOD/raptor/suitesparse/cs_ereach.c`.

use super::Cs;

/// C `cs_ereach`: finds the Cholesky nonzero pattern for one column from an etree.
pub fn cs_ereach(matrix: &Cs, column: usize, parent: &[Option<usize>]) -> Option<Vec<usize>> {
    if !matrix.is_csc()
        || column >= matrix.columns
        || parent.len() < matrix.columns
        || matrix.column_pointers.len() < matrix.columns + 1
    {
        return None;
    }
    let mut marked = vec![false; matrix.columns];
    marked[column] = true;
    let mut pattern = Vec::new();
    for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
        let mut node = *matrix.row_indices.get(entry)?;
        if node > column {
            continue;
        }
        let mut path = Vec::new();
        while !*marked.get(node)? {
            path.push(node);
            marked[node] = true;
            node = parent[node]?;
        }
        for node in path {
            pattern.push(node);
        }
    }
    Some(pattern)
}

#[cfg(test)]
mod tests {
    use super::cs_ereach;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn ereach_returns_source_stack_order() {
        let matrix = Cs {
            nzmax: 3,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 0, 0, 3],
            row_indices: vec![0, 1, 2],
            values: vec![1.; 3],
            nz: -1,
        };
        assert_eq!(
            cs_ereach(&matrix, 2, &[Some(1), Some(2), None]),
            Some(vec![0, 1])
        );
    }
}
