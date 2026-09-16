//! Translation of `IMOD/raptor/suitesparse/cs_etree.c`.

use super::Cs;

/// C `cs_etree`: computes the elimination tree of `triu(A)` or implicit `A' * A`.
pub fn cs_etree(matrix: &Cs, ata: bool) -> Option<Vec<Option<usize>>> {
    if !matrix.is_csc() || matrix.column_pointers.len() < matrix.columns + 1 {
        return None;
    }
    let entries = matrix.column_pointers[matrix.columns];
    if entries > matrix.row_indices.len() {
        return None;
    }
    let mut parent = vec![None; matrix.columns];
    let mut ancestor = vec![None; matrix.columns];
    let mut previous = if ata {
        Some(vec![None; matrix.rows])
    } else {
        None
    };
    for column in 0..matrix.columns {
        for entry in matrix.column_pointers[column]..matrix.column_pointers[column + 1] {
            let row = matrix.row_indices[entry];
            if row >= matrix.rows {
                return None;
            }
            let mut node = if let Some(previous) = previous.as_ref() {
                previous[row]
            } else {
                Some(row)
            };
            while let Some(current) = node.filter(|&current| current < column) {
                let next = ancestor[current];
                ancestor[current] = Some(column);
                if next.is_none() {
                    parent[current] = Some(column);
                }
                node = next;
            }
            if let Some(previous) = previous.as_mut() {
                previous[row] = Some(column);
            }
        }
    }
    Some(parent)
}

#[cfg(test)]
mod tests {
    use super::cs_etree;
    use crate::imod::raptor::suitesparse::Cs;
    #[test]
    fn etree_matches_csparse_path_compression() {
        let matrix = Cs {
            nzmax: 4,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 1, 3, 4],
            row_indices: vec![0, 0, 1, 1],
            values: vec![1.; 4],
            nz: -1,
        };
        assert_eq!(cs_etree(&matrix, false), Some(vec![Some(1), Some(2), None]));
    }
}
