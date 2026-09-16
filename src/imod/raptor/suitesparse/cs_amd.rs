//! Translation of `IMOD/raptor/suitesparse/cs_amd.c`.

use super::Cs;
use super::cs_add::cs_add;
use super::cs_multiply::cs_multiply;
use super::cs_transpose::cs_transpose;

fn flip(value: i64) -> i64 {
    -value - 2
}

/// C `cs_wclear`: clear AMD element marks before the mark counter wraps.
fn cs_wclear(mark: i64, lemax: i64, workspace: &mut [i64], count: usize) -> i64 {
    if mark < 2 || mark.checked_add(lemax).is_none_or(|value| value < 0) {
        for value in &mut workspace[..count] {
            if *value != 0 {
                *value = 1;
            }
        }
        2
    } else {
        mark
    }
}

/// C `cs_diag`: retain an entry only when it is off the diagonal.
fn cs_diag(row: usize, column: usize, _value: f64) -> bool {
    row != column
}

/// C `cs_amd`: approximate minimum-degree ordering.
///
/// Order 1 builds `A + A'`, order 2 uses the LU normal-equation graph, and
/// order 3 uses the QR normal-equation graph.  Like C CSparse, order zero is
/// represented by no permutation and therefore returns `None`.
pub fn cs_amd(order: i32, matrix: &Cs) -> Option<Vec<usize>> {
    if !matrix.is_csc() || !(1..=3).contains(&order) {
        return None;
    }
    let n = matrix.columns;
    if n == 0 {
        return Some(Vec::new());
    }
    let mut transpose = cs_transpose(matrix, false)?;
    let mut graph = if order == 1 && matrix.rows == n {
        cs_add(matrix, &transpose, 0.0, 0.0)?
    } else if order == 2 {
        let dense = n
            .saturating_sub(2)
            .min((10.0 * (n as f64).sqrt()).max(16.0) as usize);
        let mut write = 0;
        for column in 0..transpose.columns {
            let start = transpose.column_pointers[column];
            let end = transpose.column_pointers[column + 1];
            transpose.column_pointers[column] = write;
            if end - start > dense {
                continue;
            }
            for index in start..end {
                transpose.row_indices[write] = transpose.row_indices[index];
                write += 1;
            }
        }
        transpose.column_pointers[transpose.columns] = write;
        transpose.row_indices.truncate(write);
        transpose.values.truncate(write);
        transpose.nzmax = write;
        let second_transpose = cs_transpose(&transpose, false)?;
        cs_multiply(&transpose, &second_transpose)?
    } else {
        cs_multiply(&transpose, matrix)?
    };
    // `cs_diag` / `cs_fkeep`: AMD uses only off-diagonal graph edges.
    let mut retained = 0;
    for column in 0..graph.columns {
        let start = graph.column_pointers[column];
        let end = graph.column_pointers[column + 1];
        graph.column_pointers[column] = retained;
        for entry in start..end {
            if cs_diag(graph.row_indices[entry], column, graph.values[entry]) {
                graph.row_indices[retained] = graph.row_indices[entry];
                retained += 1;
            }
        }
    }
    graph.column_pointers[n] = retained;
    graph.row_indices.truncate(retained);
    graph.values.truncate(retained);
    graph.nzmax = retained;

    let mut cp: Vec<i64> = graph
        .column_pointers
        .iter()
        .map(|&value| value as i64)
        .collect();
    let mut ci: Vec<i64> = graph
        .row_indices
        .iter()
        .map(|&value| value as i64)
        .collect();
    let cnz = ci.len();
    let elbow = cnz.checked_add(cnz / 5)?.checked_add(2 * n)?;
    ci.resize(elbow, 0);
    let mut result = vec![-1_i64; n + 1];
    let mut work = vec![0_i64; 8 * (n + 1)];
    let (len, rest) = work.split_at_mut(n + 1);
    let (nv, rest) = rest.split_at_mut(n + 1);
    let (next, rest) = rest.split_at_mut(n + 1);
    let (head, rest) = rest.split_at_mut(n + 1);
    let (elen, rest) = rest.split_at_mut(n + 1);
    let (degree, rest) = rest.split_at_mut(n + 1);
    let (w, hhead) = rest.split_at_mut(n + 1);
    let mut last = vec![-1_i64; n + 1];
    let mut current_nz = cnz;
    let mut lemax = 0_i64;
    let mut nel = 0_i64;
    let mut mark = cs_wclear(0, 0, w, n);
    for node in 0..=n {
        head[node] = -1;
        next[node] = -1;
        hhead[node] = -1;
        last[node] = -1;
        nv[node] = 1;
        w[node] = 1;
        elen[node] = 0;
        degree[node] = if node < n { cp[node + 1] - cp[node] } else { 0 };
        len[node] = degree[node];
    }
    elen[n] = -2;
    cp[n] = -1;
    w[n] = 0;
    let dense = (n.saturating_sub(2)).min((10.0 * (n as f64).sqrt()).max(16.0) as usize) as i64;
    for node in 0..n {
        let d = degree[node];
        if d == 0 {
            elen[node] = -2;
            nel += 1;
            cp[node] = -1;
            w[node] = 0;
        } else if d > dense {
            nv[node] = 0;
            elen[node] = -1;
            nel += 1;
            cp[node] = flip(n as i64);
            nv[n] += 1;
        } else {
            let d = d as usize;
            if head[d] != -1 {
                last[head[d] as usize] = node as i64;
            }
            next[node] = head[d];
            head[d] = node as i64;
        }
    }
    let mut minimum_degree = 0usize;
    while nel < n as i64 {
        while minimum_degree < n && head[minimum_degree] == -1 {
            minimum_degree += 1;
        }
        if minimum_degree == n {
            return None;
        }
        let mut k = head[minimum_degree] as usize;
        if next[k] != -1 {
            last[next[k] as usize] = -1;
        }
        head[minimum_degree] = next[k];
        let elenk = elen[k];
        let mut nvk = nv[k];
        nel += nvk;
        if elenk > 0 && current_nz + minimum_degree >= ci.len() {
            for node in 0..n {
                let pointer = cp[node];
                if pointer >= 0 {
                    let pointer = pointer as usize;
                    cp[node] = ci[pointer];
                    ci[pointer] = flip(node as i64);
                }
            }
            let mut destination = 0;
            let mut source = 0;
            while source < current_nz {
                let node = flip(ci[source]);
                source += 1;
                if node >= 0 {
                    let node = node as usize;
                    ci[destination] = cp[node];
                    destination += 1;
                    cp[node] = (destination - 1) as i64;
                    for _ in 0..len[node] - 1 {
                        ci[destination] = ci[source];
                        destination += 1;
                        source += 1;
                    }
                }
            }
            current_nz = destination;
        }
        let mut dk = 0_i64;
        nv[k] = -nvk;
        let mut pointer = cp[k] as usize;
        let pk1 = if elenk == 0 { pointer } else { current_nz };
        let mut pk2 = pk1;
        for part in 1..=(elenk + 1) as usize {
            let (element, mut element_pointer, length) = if part > elenk as usize {
                (k, pointer, len[k] - elenk)
            } else {
                let element = ci[pointer] as usize;
                pointer += 1;
                (element, cp[element] as usize, len[element])
            };
            for _ in 0..length {
                let node = ci[element_pointer] as usize;
                element_pointer += 1;
                let node_weight = nv[node];
                if node_weight <= 0 {
                    continue;
                }
                dk += node_weight;
                nv[node] = -node_weight;
                ci[pk2] = node as i64;
                pk2 += 1;
                if next[node] != -1 {
                    last[next[node] as usize] = last[node];
                }
                if last[node] != -1 {
                    next[last[node] as usize] = next[node];
                } else {
                    head[degree[node] as usize] = next[node];
                }
            }
            if element != k {
                cp[element] = flip(k as i64);
                w[element] = 0;
            }
        }
        if elenk != 0 {
            current_nz = pk2;
        }
        degree[k] = dk;
        cp[k] = pk1 as i64;
        len[k] = (pk2 - pk1) as i64;
        elen[k] = -2;
        mark = cs_wclear(mark, lemax, w, n);
        for pk in pk1..pk2 {
            let node = ci[pk] as usize;
            let element_length = elen[node];
            if element_length <= 0 {
                continue;
            }
            let node_weight = -nv[node];
            let weighted_mark = mark - node_weight;
            let stop = cp[node] + element_length;
            for position in cp[node]..stop {
                let element = ci[position as usize] as usize;
                if w[element] >= mark {
                    w[element] -= node_weight;
                } else if w[element] != 0 {
                    w[element] = degree[element] + weighted_mark;
                }
            }
        }
        for pk in pk1..pk2 {
            let node = ci[pk] as usize;
            let p1 = cp[node] as usize;
            let p2 = (cp[node] + elen[node] - 1) as usize;
            let mut pn = p1;
            let mut hash = 0_i64;
            let mut d = 0_i64;
            for position in p1..=p2 {
                let element = ci[position] as usize;
                if w[element] != 0 {
                    let external_degree = w[element] - mark;
                    if external_degree > 0 {
                        d += external_degree;
                        ci[pn] = element as i64;
                        pn += 1;
                        hash += element as i64;
                    } else {
                        cp[element] = flip(k as i64);
                        w[element] = 0;
                    }
                }
            }
            elen[node] = (pn - p1 + 1) as i64;
            let p3 = pn;
            let p4 = p1 + len[node] as usize;
            for position in (p2 + 1)..p4 {
                let neighbor = ci[position] as usize;
                let neighbor_weight = nv[neighbor];
                if neighbor_weight <= 0 {
                    continue;
                }
                d += neighbor_weight;
                ci[pn] = neighbor as i64;
                pn += 1;
                hash += neighbor as i64;
            }
            if d == 0 {
                cp[node] = flip(k as i64);
                let node_weight = -nv[node];
                dk -= node_weight;
                nvk += node_weight;
                nel += node_weight;
                nv[node] = 0;
                elen[node] = -1;
            } else {
                degree[node] = degree[node].min(d);
                ci[pn] = ci[p3];
                ci[p3] = ci[p1];
                ci[p1] = k as i64;
                len[node] = (pn - p1 + 1) as i64;
                let hash = (hash as usize) % n;
                next[node] = hhead[hash];
                hhead[hash] = node as i64;
                last[node] = hash as i64;
            }
        }
        degree[k] = dk;
        lemax = lemax.max(dk);
        mark = cs_wclear(mark.checked_add(lemax)?, lemax, w, n);
        for pk in pk1..pk2 {
            let mut node = ci[pk] as usize;
            if nv[node] >= 0 {
                continue;
            }
            let hash = last[node] as usize;
            node = hhead[hash] as usize;
            hhead[hash] = -1;
            while node != usize::MAX && next[node] != -1 {
                let length = len[node];
                let element_length = elen[node];
                for position in (cp[node] + 1)..=(cp[node] + length - 1) {
                    w[ci[position as usize] as usize] = mark;
                }
                let mut previous = node;
                let mut candidate = next[node];
                while candidate != -1 {
                    let candidate_index = candidate as usize;
                    let mut equal =
                        len[candidate_index] == length && elen[candidate_index] == element_length;
                    for position in (cp[candidate_index] + 1)..=(cp[candidate_index] + length - 1) {
                        if equal && w[ci[position as usize] as usize] != mark {
                            equal = false;
                        }
                    }
                    if equal {
                        cp[candidate_index] = flip(node as i64);
                        nv[node] += nv[candidate_index];
                        nv[candidate_index] = 0;
                        elen[candidate_index] = -1;
                        candidate = next[candidate_index];
                        next[previous] = candidate;
                    } else {
                        previous = candidate_index;
                        candidate = next[candidate_index];
                    }
                }
                mark = mark.checked_add(1)?;
                node = next[node].try_into().unwrap_or(usize::MAX);
            }
        }
        let mut write = pk1;
        for pk in pk1..pk2 {
            let node = ci[pk] as usize;
            let node_weight = -nv[node];
            if node_weight <= 0 {
                continue;
            }
            nv[node] = node_weight;
            let d = (degree[node] + dk - node_weight).min(n as i64 - nel - node_weight);
            if head[d as usize] != -1 {
                last[head[d as usize] as usize] = node as i64;
            }
            next[node] = head[d as usize];
            last[node] = -1;
            head[d as usize] = node as i64;
            minimum_degree = minimum_degree.min(d as usize);
            degree[node] = d;
            ci[write] = node as i64;
            write += 1;
        }
        nv[k] = nvk;
        len[k] = (write - pk1) as i64;
        if len[k] == 0 {
            cp[k] = -1;
            w[k] = 0;
        }
        if elenk != 0 {
            current_nz = write;
        }
    }
    for node in 0..n {
        cp[node] = flip(cp[node]);
    }
    head.fill(-1);
    for node in (0..=n).rev() {
        if nv[node] <= 0 {
            let parent = cp[node] as usize;
            next[node] = head[parent];
            head[parent] = node as i64;
        }
    }
    for element in (0..=n).rev() {
        if nv[element] > 0 && cp[element] != -1 {
            let parent = cp[element] as usize;
            next[element] = head[parent];
            head[parent] = element as i64;
        }
    }
    let mut out = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for root in 0..=n {
        if cp[root] != -1 {
            continue;
        }
        stack.push((root, head[root]));
        while let Some((node, child)) = stack.pop() {
            if child == -1 {
                if node < n {
                    out.push(node);
                }
            } else {
                stack.push((node, next[child as usize]));
                stack.push((child as usize, head[child as usize]));
            }
        }
    }
    (out.len() == n).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::{cs_amd, cs_diag, cs_wclear};
    use crate::imod::raptor::suitesparse::Cs;

    #[test]
    fn amd_returns_a_permutation_for_each_supported_graph() {
        let matrix = Cs {
            nzmax: 7,
            rows: 3,
            columns: 3,
            column_pointers: vec![0, 2, 5, 7],
            row_indices: vec![0, 1, 0, 1, 2, 1, 2],
            values: vec![1.0; 7],
            nz: -1,
        };
        for order in 1..=3 {
            let mut permutation = cs_amd(order, &matrix).unwrap();
            permutation.sort_unstable();
            assert_eq!(permutation, vec![0, 1, 2]);
        }
        assert_eq!(cs_amd(0, &matrix), None);
    }

    #[test]
    fn amd_mark_clear_and_diagonal_filter_follow_c_helpers() {
        let mut marks = [0, 5, -4];
        assert_eq!(cs_wclear(0, 0, &mut marks, 3), 2);
        assert_eq!(marks, [0, 1, 1]);
        assert_eq!(cs_wclear(5, 3, &mut marks, 3), 5);
        assert!(cs_diag(0, 1, 4.0));
        assert!(!cs_diag(1, 1, 4.0));
    }
}
