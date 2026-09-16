//! Owned translation of `IMOD/raptor/lasik/svl/lib/pgm/svlGraphUtils.{h,cpp}`.

use std::collections::BTreeSet;

/// C++ `svlEdge`.
pub type SvlEdge = (usize, usize);

/// C++ `svlWeightedEdge`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvlWeightedEdge {
    pub node_a: isize,
    pub node_b: isize,
    pub w_ab: f64,
    pub w_ba: f64,
}

impl Default for SvlWeightedEdge {
    fn default() -> Self {
        Self {
            node_a: -1,
            node_b: -1,
            w_ab: 0.0,
            w_ba: 0.0,
        }
    }
}

impl SvlWeightedEdge {
    pub fn new(node_a: isize, node_b: isize, w_ab: f64, w_ba: f64) -> Self {
        Self {
            node_a,
            node_b,
            w_ab,
            w_ba,
        }
    }
}

/// C++ `svlTriangulationHeuristic`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SvlTriangulationHeuristic {
    MaxCardSearch,
    MinFillIn,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SvlGraphUtilsError {
    InvalidNode,
    MismatchedWeights,
    EmptyGraph,
    NoPath,
}

/// Source `minSpanningTree`, Kruskal's minimum spanning forest.
pub fn min_spanning_tree(
    num_nodes: usize,
    edges: &[SvlEdge],
    weights: &[f64],
) -> Result<Vec<SvlEdge>, SvlGraphUtilsError> {
    if num_nodes == 0 {
        return Err(SvlGraphUtilsError::EmptyGraph);
    }
    if edges
        .iter()
        .any(|&(first, second)| first >= num_nodes || second >= num_nodes)
    {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    if edges.len() != weights.len() {
        return Err(SvlGraphUtilsError::MismatchedWeights);
    }
    let mut ordered: Vec<_> = edges.iter().copied().zip(weights.iter().copied()).collect();
    ordered.sort_by(|(_, first), (_, second)| first.total_cmp(second));
    let mut forest: Vec<_> = (0..num_nodes).collect();
    let mut tree = Vec::new();
    for ((first, second), _) in ordered {
        let first_forest = forest[first];
        let second_forest = forest[second];
        if first_forest == second_forest {
            continue;
        }
        tree.push((first, second));
        for value in &mut forest {
            if *value == second_forest {
                *value = first_forest;
            }
        }
    }
    Ok(tree)
}

/// Result of C++ `maxCardinalitySearch`: either a perfect elimination order
/// or the three vertices that violate chordality.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxCardinalitySearch {
    PerfectOrder(Vec<usize>),
    OffendingNodes([usize; 3]),
}

/// Source `maxCardinalitySearch`.
pub fn max_cardinality_search(
    num_nodes: usize,
    edges: &[SvlEdge],
    start_node: Option<usize>,
) -> Result<MaxCardinalitySearch, SvlGraphUtilsError> {
    if num_nodes == 0 {
        return Err(SvlGraphUtilsError::EmptyGraph);
    }
    if edges
        .iter()
        .any(|&(first, second)| first >= num_nodes || second >= num_nodes)
    {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    if start_node.is_some_and(|node| node >= num_nodes) {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    let mut neighbors = vec![BTreeSet::new(); num_nodes];
    for &(first, second) in edges {
        neighbors[first].insert(second);
        neighbors[second].insert(first);
    }
    let mut next = start_node.unwrap_or_else(|| {
        (1..num_nodes).fold(0, |best, node| {
            if neighbors[node].len() > neighbors[best].len() {
                node
            } else {
                best
            }
        })
    });
    let mut ordering = vec![None; num_nodes];
    let mut weights = vec![0usize; num_nodes];
    for remaining in (0..num_nodes).rev() {
        ordering[next] = Some(remaining);
        if remaining == 0 {
            break;
        }
        for &node in &neighbors[next] {
            if ordering[node].is_none() {
                weights[node] += 1;
            }
        }
        next = (0..num_nodes)
            .filter(|&node| ordering[node].is_none())
            .max_by_key(|&node| weights[node])
            .expect("unlabeled node exists while vertices remain");
        let labeled: Vec<_> = neighbors[next]
            .iter()
            .copied()
            .filter(|&node| ordering[node].is_some())
            .collect();
        for (index, &first) in labeled.iter().enumerate() {
            for &second in &labeled[index + 1..] {
                if !neighbors[first].contains(&second) {
                    return Ok(MaxCardinalitySearch::OffendingNodes([next, first, second]));
                }
            }
        }
    }
    let mut perfect_order = vec![0; num_nodes];
    for (node, order) in ordering.into_iter().enumerate() {
        perfect_order[order.expect("every node is ordered")] = node;
    }
    Ok(MaxCardinalitySearch::PerfectOrder(perfect_order))
}

/// Source `triangulateGraph`.
pub fn triangulate_graph(
    num_nodes: usize,
    edges: &mut Vec<SvlEdge>,
    method: SvlTriangulationHeuristic,
) -> Result<(), SvlGraphUtilsError> {
    if num_nodes == 0 {
        return Err(SvlGraphUtilsError::EmptyGraph);
    }
    if edges
        .iter()
        .any(|&(first, second)| first >= num_nodes || second >= num_nodes)
    {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    match method {
        SvlTriangulationHeuristic::MaxCardSearch => loop {
            match max_cardinality_search(num_nodes, edges, None)? {
                MaxCardinalitySearch::PerfectOrder(_) => break,
                MaxCardinalitySearch::OffendingNodes([_, first, second]) => {
                    edges.push((first, second))
                }
            }
        },
        SvlTriangulationHeuristic::MinFillIn => {
            let mut neighbors = vec![BTreeSet::new(); num_nodes];
            for &(first, second) in edges.iter() {
                neighbors[first].insert(second);
                neighbors[second].insert(first);
            }
            let mut eliminated = vec![false; num_nodes];
            for _ in 0..num_nodes {
                let node = (0..num_nodes)
                    .filter(|&node| !eliminated[node])
                    .min_by_key(|&node| {
                        let active: Vec<_> = neighbors[node]
                            .iter()
                            .copied()
                            .filter(|&neighbor| !eliminated[neighbor])
                            .collect();
                        active
                            .iter()
                            .enumerate()
                            .map(|(index, first)| {
                                active[index + 1..]
                                    .iter()
                                    .filter(|second| !neighbors[*first].contains(second))
                                    .count()
                            })
                            .sum::<usize>()
                    })
                    .expect("one non-eliminated node exists");
                eliminated[node] = true;
                let active: Vec<_> = neighbors[node]
                    .iter()
                    .copied()
                    .filter(|&neighbor| !eliminated[neighbor])
                    .collect();
                for (index, &first) in active.iter().enumerate() {
                    for &second in &active[index + 1..] {
                        if neighbors[first].insert(second) {
                            neighbors[second].insert(first);
                            edges.push((first, second));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Source `variableEliminationCliques`.
pub fn variable_elimination_cliques(
    edges: &[SvlEdge],
    node_order: &[usize],
) -> Vec<BTreeSet<usize>> {
    let mut cliques = Vec::new();
    let mut remaining = edges.to_vec();
    for &node in node_order {
        let mut clique = BTreeSet::from([node]);
        remaining.retain(|&(first, second)| {
            if first == node {
                clique.insert(second);
                false
            } else if second == node {
                clique.insert(first);
                false
            } else {
                true
            }
        });
        if !cliques
            .iter()
            .rev()
            .any(|existing: &BTreeSet<usize>| clique.is_subset(existing))
        {
            cliques.push(clique);
        }
    }
    cliques
}

/// Source `allShortestPaths`, Floyd-Warshall over an undirected weighted graph.
pub fn all_shortest_paths(
    num_nodes: usize,
    edges: &[SvlEdge],
    weights: &[f64],
) -> Result<Vec<Vec<f64>>, SvlGraphUtilsError> {
    if num_nodes == 0 {
        return Err(SvlGraphUtilsError::EmptyGraph);
    }
    if edges
        .iter()
        .any(|&(first, second)| first >= num_nodes || second >= num_nodes)
    {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    if edges.len() != weights.len() {
        return Err(SvlGraphUtilsError::MismatchedWeights);
    }
    let mut distances = vec![vec![f64::INFINITY; num_nodes]; num_nodes];
    for (&(first, second), &weight) in edges.iter().zip(weights) {
        distances[first][second] = weight;
        distances[second][first] = weight;
    }
    for (node, row) in distances.iter_mut().enumerate() {
        row[node] = 0.0;
    }
    for middle in 0..num_nodes {
        for source in 0..num_nodes {
            for sink in 0..num_nodes {
                distances[source][sink] = distances[source][sink]
                    .min(distances[source][middle] + distances[middle][sink]);
            }
        }
    }
    Ok(distances)
}

/// Source `shortestPath`. Like the source, the returned path ends at the
/// predecessor of `sink` rather than including `sink` itself.
pub fn shortest_path(
    num_nodes: usize,
    edges: &[SvlEdge],
    weights: &[f64],
    source: usize,
    sink: usize,
) -> Result<Vec<usize>, SvlGraphUtilsError> {
    if num_nodes == 0 {
        return Err(SvlGraphUtilsError::EmptyGraph);
    }
    if edges
        .iter()
        .any(|&(first, second)| first >= num_nodes || second >= num_nodes)
    {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    if source >= num_nodes || sink >= num_nodes {
        return Err(SvlGraphUtilsError::InvalidNode);
    }
    if edges.len() != weights.len() {
        return Err(SvlGraphUtilsError::MismatchedWeights);
    }
    let mut neighbors = vec![Vec::new(); num_nodes];
    for (&(first, second), &weight) in edges.iter().zip(weights) {
        neighbors[first].push((second, weight));
        neighbors[second].push((first, weight));
    }
    let mut distances = vec![f64::INFINITY; num_nodes];
    let mut previous = vec![None; num_nodes];
    let mut visited = vec![false; num_nodes];
    distances[source] = 0.0;
    for _ in 0..num_nodes {
        let Some(current) = (0..num_nodes)
            .filter(|&node| !visited[node])
            .min_by(|&first, &second| distances[first].total_cmp(&distances[second]))
        else {
            break;
        };
        if distances[current].is_infinite() || current == sink {
            break;
        }
        visited[current] = true;
        for &(neighbor, weight) in &neighbors[current] {
            let candidate = distances[current] + weight;
            if candidate < distances[neighbor] {
                distances[neighbor] = candidate;
                previous[neighbor] = Some(current);
            }
        }
    }
    if source == sink {
        return Ok(vec![source]);
    }
    let mut path = Vec::new();
    let mut node = previous[sink].ok_or(SvlGraphUtilsError::NoPath)?;
    while node != source {
        path.push(node);
        node = previous[node].ok_or(SvlGraphUtilsError::NoPath)?;
    }
    path.push(source);
    path.reverse();
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forest_shortest_paths_and_source_path_layout_match() {
        let edges = [(0, 1), (1, 2), (0, 2), (2, 3)];
        let weights = [1.0, 2.0, 4.0, 1.0];
        assert_eq!(
            min_spanning_tree(4, &edges, &weights).unwrap(),
            vec![(0, 1), (2, 3), (1, 2)]
        );
        assert_eq!(all_shortest_paths(4, &edges, &weights).unwrap()[0][3], 4.0);
        assert_eq!(
            shortest_path(4, &edges, &weights, 0, 3).unwrap(),
            vec![0, 1, 2]
        );
    }
    #[test]
    fn chordal_search_triangulation_and_cliques_work() {
        let mut cycle = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
        assert!(matches!(
            max_cardinality_search(4, &cycle, None).unwrap(),
            MaxCardinalitySearch::OffendingNodes(_)
        ));
        triangulate_graph(4, &mut cycle, SvlTriangulationHeuristic::MinFillIn).unwrap();
        assert!(matches!(
            max_cardinality_search(4, &cycle, None).unwrap(),
            MaxCardinalitySearch::PerfectOrder(_)
        ));
        let cliques = variable_elimination_cliques(&cycle, &[0, 1, 2, 3]);
        assert!(cliques.iter().any(|clique| clique.len() >= 3));
    }
    #[test]
    fn weighted_edge_defaults_match_header() {
        assert_eq!(
            SvlWeightedEdge::default(),
            SvlWeightedEdge {
                node_a: -1,
                node_b: -1,
                w_ab: 0.0,
                w_ba: 0.0
            }
        );
    }
}
