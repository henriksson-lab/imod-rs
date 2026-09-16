//! Translation of `IMOD/raptor/suitesparse/cs_updown.c`.

use super::Cs;

/// C `cs_updown`: sparse rank-one Cholesky update or downdate.
///
/// Updates `lower * lower'` by `sigma * column(C) * column(C)'`, where
/// `sigma` is positive for an update and negative for a downdate.  As in the
/// source, only the first CSC column of `update` participates and a failed
/// downdate may have already changed the factor along the reached path.
pub fn cs_updown(lower: &mut Cs, sigma: i32, update: &Cs, parent: &[Option<usize>]) -> bool {
    if !lower.is_csc()
        || !update.is_csc()
        || lower.rows != lower.columns
        || lower.column_pointers.len() != lower.columns + 1
        || update.column_pointers.len() < 2
        || parent.len() != lower.columns
        || parent.iter().flatten().any(|&node| node >= lower.columns)
    {
        return false;
    }
    let n = lower.columns;
    let update_end = update.column_pointers[1];
    if update.column_pointers[0] > update_end
        || update_end > update.row_indices.len()
        || update_end > update.values.len()
        || lower.column_pointers[n] > lower.row_indices.len()
        || lower.column_pointers[n] > lower.values.len()
        || lower
            .column_pointers
            .windows(2)
            .any(|offsets| offsets[0] > offsets[1])
    {
        return false;
    }
    let update_start = update.column_pointers[0];
    if update_start == update_end {
        return true;
    }
    let mut first = n;
    for entry in update_start..update_end {
        let row = update.row_indices[entry];
        if row >= n {
            return false;
        }
        first = first.min(row);
    }
    let mut workspace = vec![0.0; n];
    let mut node = Some(first);
    // The parent structure is an elimination forest.  Detect malformed cycles
    // before executing the source's unbounded parent walk.
    for _ in 0..n {
        let Some(current) = node else { break };
        workspace[current] = 0.0;
        node = parent[current];
    }
    if node.is_some() {
        return false;
    }
    for entry in update_start..update_end {
        workspace[update.row_indices[entry]] = update.values[entry];
    }
    let mut beta = 1.0;
    let mut beta_squared = 1.0;
    node = Some(first);
    while let Some(column) = node {
        let diagonal = lower.column_pointers[column];
        if diagonal >= lower.values.len() || lower.row_indices[diagonal] != column {
            return false;
        }
        let alpha = workspace[column] / lower.values[diagonal];
        beta_squared = beta * beta + (sigma as f64) * alpha * alpha;
        if beta_squared <= 0.0 {
            break;
        }
        beta_squared = beta_squared.sqrt();
        let delta = if sigma > 0 {
            beta / beta_squared
        } else {
            beta_squared / beta
        };
        let gamma = (sigma as f64) * alpha / (beta_squared * beta);
        lower.values[diagonal] = delta * lower.values[diagonal]
            + if sigma > 0 {
                gamma * workspace[column]
            } else {
                0.0
            };
        beta = beta_squared;
        for entry in diagonal + 1..lower.column_pointers[column + 1] {
            let row = lower.row_indices[entry];
            if row >= n {
                return false;
            }
            let old = workspace[row];
            let new = old - alpha * lower.values[entry];
            workspace[row] = new;
            lower.values[entry] =
                delta * lower.values[entry] + gamma * if sigma > 0 { old } else { new };
        }
        node = parent[column];
    }
    beta_squared > 0.0
}

#[cfg(test)]
mod tests {
    use super::cs_updown;
    use crate::imod::raptor::suitesparse::Cs;

    fn identity_factor() -> Cs {
        Cs {
            nzmax: 2,
            rows: 2,
            columns: 2,
            column_pointers: vec![0, 1, 2],
            row_indices: vec![0, 1],
            values: vec![1.0, 1.0],
            nz: -1,
        }
    }

    #[test]
    fn update_and_downdate_follow_cholesky_rank_one_formula() {
        let update = Cs {
            nzmax: 1,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![1.0],
            nz: -1,
        };
        let mut factor = identity_factor();
        assert!(cs_updown(&mut factor, 1, &update, &[Some(1), None]));
        assert!((factor.values[0] - 2.0_f64.sqrt()).abs() < 1.0e-14);
        assert_eq!(factor.values[1], 1.0);

        let mut factor = identity_factor();
        let mut downdate = update.clone();
        downdate.values[0] = 0.5;
        assert!(cs_updown(&mut factor, -1, &downdate, &[Some(1), None]));
        assert!((factor.values[0] - 0.75_f64.sqrt()).abs() < 1.0e-14);
    }

    #[test]
    fn failed_downdate_and_empty_update_match_source_status() {
        let update = Cs {
            nzmax: 1,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 1],
            row_indices: vec![0],
            values: vec![2.0],
            nz: -1,
        };
        let mut factor = identity_factor();
        assert!(!cs_updown(&mut factor, -1, &update, &[Some(1), None]));
        let empty = Cs {
            nzmax: 0,
            rows: 2,
            columns: 1,
            column_pointers: vec![0, 0],
            row_indices: vec![],
            values: vec![],
            nz: -1,
        };
        assert!(cs_updown(&mut factor, 1, &empty, &[Some(1), None]));
    }
}
