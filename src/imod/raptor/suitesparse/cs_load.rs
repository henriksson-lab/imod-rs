//! Translation of `IMOD/raptor/suitesparse/cs_load.c`.

use std::io::BufRead;

use super::Cs;
use super::cs_entry::cs_entry;

/// C `cs_load`: reads whitespace-separated `row column value` triplets until parsing stops.
pub fn cs_load(reader: &mut impl BufRead) -> std::io::Result<Cs> {
    let mut matrix = Cs {
        nzmax: 1,
        nz: 0,
        ..Cs::default()
    };
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let mut fields = line.split_whitespace();
        let Some(row) = fields.next().and_then(|field| field.parse::<usize>().ok()) else {
            break;
        };
        let Some(column) = fields.next().and_then(|field| field.parse::<usize>().ok()) else {
            break;
        };
        let Some(value) = fields.next().and_then(|field| field.parse::<f64>().ok()) else {
            break;
        };
        if !cs_entry(&mut matrix, row, column, value) {
            break;
        }
    }
    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::cs_load;
    use std::io::Cursor;
    #[test]
    fn load_reads_triplets_until_the_first_non_triplet() {
        let mut input = Cursor::new(b"1 2 3.5\n0 1 -2\nstop\n3 4 5\n");
        let matrix = cs_load(&mut input).unwrap();
        assert_eq!(matrix.nz, 2);
        assert_eq!(&matrix.row_indices[..2], &[1, 0]);
        assert_eq!(&matrix.column_pointers[..2], &[2, 1]);
        assert_eq!(&matrix.values[..2], &[3.5, -2.0]);
    }
}
