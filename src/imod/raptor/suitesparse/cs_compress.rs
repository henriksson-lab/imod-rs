//! Translation of `IMOD/raptor/suitesparse/cs_compress.c`.
use super::Cs;

/// C `cs_compress`: converts triplet storage to compressed-column storage.
pub fn cs_compress(triplet: &Cs) -> Option<Cs> {
    if !triplet.is_triplet() || triplet.nz < 0 {
        return None;
    }
    let entries = triplet.nz as usize;
    if entries > triplet.column_pointers.len()
        || entries > triplet.row_indices.len()
        || entries > triplet.values.len()
    {
        return None;
    }
    let mut counts = vec![0_usize; triplet.columns];
    for &column in &triplet.column_pointers[..entries] {
        *counts.get_mut(column)? += 1;
    }
    let mut pointers = vec![0; triplet.columns + 1];
    for column in 0..triplet.columns {
        pointers[column + 1] = pointers[column] + counts[column];
    }
    let mut insertion = pointers[..triplet.columns].to_vec();
    let mut rows = vec![0; entries];
    let mut values = vec![0.0; entries];
    for entry in 0..entries {
        let column = triplet.column_pointers[entry];
        let target = insertion[column];
        rows[target] = triplet.row_indices[entry];
        values[target] = triplet.values[entry];
        insertion[column] += 1;
    }
    Some(Cs {
        nzmax: entries,
        rows: triplet.rows,
        columns: triplet.columns,
        column_pointers: pointers,
        row_indices: rows,
        values,
        nz: -1,
    })
}
